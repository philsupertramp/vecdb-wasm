#!/usr/bin/env python3
"""
vROM Builder — Pre-computed Vector Read-Only Memory packages for VecDB-WASM.

Produces HNSW index JSON files that are directly loadable by VectorDB.load()
in the browser, plus a chunks.json metadata payload and manifest.json.

Architecture:
  1. Fetch docs (markdown) from source
  2. Section-aware chunking with linked-list pointers (prev/next)
  3. Embed with sentence-transformers (all-MiniLM-L6-v2)
  4. Build HNSW graph matching VecDB-WASM's Rust serde schema
  5. Serialize to JSON + upload to HF Hub as a dataset repo
"""

import json
import math
import hashlib
import random
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ─── HNSW Index Builder (mirrors VecDB-WASM Rust impl) ───────────────────────

class HnswConfig:
    """Mirrors the Rust HnswConfig struct exactly."""
    def __init__(self, m=16, ef_construction=128, ef_search=40, metric="Cosine"):
        self.m = m
        self.m_max0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.level_multiplier = 1.0 / math.log(m)
        self.metric = metric  # "Cosine", "Euclidean", "DotProduct"

    def to_dict(self):
        return {
            "m": self.m,
            "m_max0": self.m_max0,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "level_multiplier": self.level_multiplier,
            "metric": self.metric,
        }


class HnswNode:
    """Mirrors the Rust HnswNode struct."""
    __slots__ = ('vector', 'neighbors', 'max_layer', 'metadata')

    def __init__(self, vector: list, max_layer: int, metadata: Optional[str] = None):
        self.vector = vector
        self.neighbors = [[] for _ in range(max_layer + 1)]
        self.max_layer = max_layer
        self.metadata = metadata

    def to_dict(self):
        return {
            "vector": self.vector,
            "neighbors": self.neighbors,
            "max_layer": self.max_layer,
            "metadata": self.metadata,
        }


def cosine_distance(a, b):
    """Cosine distance = 1 - cosine_similarity. Vectors assumed normalized."""
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot


class HnswIndex:
    """
    Pure-Python HNSW index that produces JSON byte-identical to VecDB-WASM's
    serde serialization. This ensures VectorDB.load() works directly.
    """

    def __init__(self, dim: int, config: HnswConfig = None, seed: int = 42):
        self.dim = dim
        self.config = config or HnswConfig()
        self.nodes: list[HnswNode] = []
        self.entry_point = None
        self.max_layer = 0
        self._rng = random.Random(seed)

    def _random_level(self) -> int:
        r = self._rng.random()
        # Clamp to avoid log(0)
        r = max(r, 1e-10)
        return int(-math.log(r) * self.config.level_multiplier)

    def _distance(self, query: list, node_id: int) -> float:
        return cosine_distance(query, self.nodes[node_id].vector)

    def _search_layer_single(self, query, entry, layer):
        current = entry
        current_dist = self._distance(query, current)
        changed = True
        while changed:
            changed = False
            node = self.nodes[current]
            if layer < len(node.neighbors):
                for neighbor in node.neighbors[layer]:
                    d = self._distance(query, neighbor)
                    if d < current_dist:
                        current = neighbor
                        current_dist = d
                        changed = True
        return current

    def _search_layer(self, query, entry_points, ef, layer):
        visited = set()
        # candidates: min-heap (closest first) — use negative dist for max-heap behavior
        # results: max-heap (farthest first)
        import heapq
        candidates = []  # (dist, id) — min-heap
        results = []     # (-dist, id) — max-heap (we negate for heapq)

        for ep in entry_points:
            if ep not in visited:
                visited.add(ep)
                d = self._distance(query, ep)
                heapq.heappush(candidates, (d, ep))
                heapq.heappush(results, (-d, ep))

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            # Check if we can stop
            if len(results) >= ef:
                worst_dist = -results[0][0]
                if current_dist > worst_dist:
                    break

            node = self.nodes[current]
            if layer < len(node.neighbors):
                for neighbor in node.neighbors[layer]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        d = self._distance(query, neighbor)
                        should_add = len(results) < ef or d < -results[0][0]
                        if should_add:
                            heapq.heappush(candidates, (d, neighbor))
                            heapq.heappush(results, (-d, neighbor))
                            if len(results) > ef:
                                heapq.heappop(results)

        # Sort by distance ascending
        result_list = [(-neg_d, nid) for neg_d, nid in results]
        result_list.sort(key=lambda x: x[0])
        return result_list  # [(dist, id), ...]

    def _shrink_connections(self, node_id, layer, max_connections):
        node = self.nodes[node_id]
        neighbors_with_dist = []
        for nid in node.neighbors[layer]:
            d = cosine_distance(node.vector, self.nodes[nid].vector)
            neighbors_with_dist.append((d, nid))
        neighbors_with_dist.sort(key=lambda x: x[0])
        node.neighbors[layer] = [nid for _, nid in neighbors_with_dist[:max_connections]]

    def insert(self, vector: list, metadata: Optional[str] = None) -> int:
        assert len(vector) == self.dim, f"Expected {self.dim}-dim, got {len(vector)}"
        new_id = len(self.nodes)
        new_level = self._random_level()
        node = HnswNode(vector, new_level, metadata)
        self.nodes.append(node)

        if len(self.nodes) == 1:
            self.entry_point = new_id
            self.max_layer = new_level
            return new_id

        ep = self.entry_point
        query = vector
        current_ep = ep

        # Traverse from top layers down to new_level + 1
        for layer in range(self.max_layer, new_level, -1):
            current_ep = self._search_layer_single(query, current_ep, layer)

        start_layer = min(new_level, self.max_layer)
        ep_set = [current_ep]

        for layer in range(start_layer, -1, -1):
            m_max = self.config.m_max0 if layer == 0 else self.config.m
            candidates = self._search_layer(query, ep_set, self.config.ef_construction, layer)
            neighbors = [nid for _, nid in candidates[:m_max]]
            self.nodes[new_id].neighbors[layer] = neighbors

            for neighbor_id in neighbors:
                nnode = self.nodes[neighbor_id]
                if layer < len(nnode.neighbors):
                    nnode.neighbors[layer].append(new_id)
                    if len(nnode.neighbors[layer]) > m_max:
                        self._shrink_connections(neighbor_id, layer, m_max)

            ep_set = [nid for _, nid in candidates[:self.config.m]]

        if new_level > self.max_layer:
            self.entry_point = new_id
            self.max_layer = new_level

        return new_id

    def search(self, query: list, k: int, ef_search: int = None) -> list:
        if ef_search is None:
            ef_search = self.config.ef_search
        assert len(query) == self.dim
        if not self.nodes:
            return []

        ep = self.entry_point
        current_ep = ep
        if self.max_layer > 0:
            for layer in range(self.max_layer, 0, -1):
                current_ep = self._search_layer_single(query, current_ep, layer)

        ef = max(ef_search, k)
        candidates = self._search_layer(query, [current_ep], ef, 0)
        return candidates[:k]  # [(dist, id), ...]

    def to_dict(self):
        """Serialize to dict matching VecDB-WASM's serde JSON schema exactly."""
        return {
            "config": self.config.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "entry_point": self.entry_point,
            "max_layer": self.max_layer,
            "dim": self.dim,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ─── Section-Aware Chunker ────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: int
    text: str
    source_file: str
    section_heading: str
    char_start: int
    char_end: int
    token_estimate: int
    prev_chunk_id: Optional[int] = None
    next_chunk_id: Optional[int] = None
    # Extra metadata for citations
    url: Optional[str] = None
    doc_title: Optional[str] = None


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def section_aware_chunk(
    text: str,
    source_file: str,
    doc_title: str = "",
    url: str = "",
    max_tokens: int = 256,
    start_chunk_id: int = 0,
) -> list[Chunk]:
    """
    Section-aware chunker for markdown documentation.

    Strategy (from research):
    - Split on markdown headings (##, ###, etc.)
    - Keep code blocks intact within chunks
    - Target 150-300 tokens per chunk (configurable)
    - Zero overlap (paper 2601.14123: overlap adds cost, no gain)
    - Linked-list pointers for context traversal
    """
    chunks = []
    chunk_id = start_chunk_id

    # Split into sections by headings
    # Pattern: split before any line starting with # (one or more)
    sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading
        heading_match = re.match(r'^(#{1,6})\s+(.*?)$', section, re.MULTILINE)
        heading = heading_match.group(2).strip() if heading_match else doc_title

        # If section is small enough, keep as one chunk
        if estimate_tokens(section) <= max_tokens:
            char_start = text.find(section)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=section,
                source_file=source_file,
                section_heading=heading,
                char_start=char_start,
                char_end=char_start + len(section),
                token_estimate=estimate_tokens(section),
                url=url,
                doc_title=doc_title,
            ))
            chunk_id += 1
            continue

        # Split large sections into paragraphs, keeping code blocks intact
        # First, protect code blocks
        code_blocks = []
        def replace_code(m):
            code_blocks.append(m.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        protected = re.sub(r'```[\s\S]*?```', replace_code, section)

        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\n+', protected)

        current_text = ""
        for para in paragraphs:
            # Restore code blocks in this paragraph
            restored_para = para
            for i, cb in enumerate(code_blocks):
                restored_para = restored_para.replace(f"__CODE_BLOCK_{i}__", cb)

            if estimate_tokens(current_text + "\n\n" + restored_para) > max_tokens and current_text:
                # Emit current chunk
                char_start = text.find(current_text.strip()) if current_text.strip() in text else 0
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=current_text.strip(),
                    source_file=source_file,
                    section_heading=heading,
                    char_start=char_start,
                    char_end=char_start + len(current_text.strip()),
                    token_estimate=estimate_tokens(current_text),
                    url=url,
                    doc_title=doc_title,
                ))
                chunk_id += 1
                current_text = restored_para
            else:
                current_text = (current_text + "\n\n" + restored_para).strip()

        # Emit remaining
        if current_text.strip():
            char_start = text.find(current_text.strip()) if current_text.strip() in text else 0
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=current_text.strip(),
                source_file=source_file,
                section_heading=heading,
                char_start=char_start,
                char_end=char_start + len(current_text.strip()),
                token_estimate=estimate_tokens(current_text),
                url=url,
                doc_title=doc_title,
            ))
            chunk_id += 1

    # Wire up linked list pointers
    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
        chunk.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None

    return chunks


# ─── vROM Builder ─────────────────────────────────────────────────────────────

@dataclass
class VromManifest:
    vrom_id: str
    version: str
    description: str
    source: str
    embedding_spec: dict
    hnsw_config: dict
    vector_count: int
    total_tokens: int
    total_chunks: int
    corpus_hash: str
    created_at: str
    chunk_strategy: dict


class VromBuilder:
    """
    Builds a vROM package: HNSW index + chunks metadata + manifest.

    The HNSW index JSON is directly loadable by VecDB-WASM's VectorDB.load().
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 128,
        hnsw_ef_search: int = 40,
        max_chunk_tokens: int = 256,
    ):
        self.model_name = model_name
        self.dim = dim
        self.max_chunk_tokens = max_chunk_tokens

        # HNSW config
        self.hnsw_config = HnswConfig(
            m=hnsw_m,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
            metric="Cosine",
        )

        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Dimension: {self.model.get_embedding_dimension()}")
        assert self.model.get_embedding_dimension() == dim

    def build(
        self,
        documents: list[dict],  # [{"text": ..., "source_file": ..., "title": ..., "url": ...}]
        vrom_id: str,
        version: str = "1.0.0",
        description: str = "",
        source: str = "",
        output_dir: str = "./vrom_output",
    ) -> Path:
        """
        Build a complete vROM package from documents.

        Returns path to output directory.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # 1. Chunk all documents
        print(f"\n=== Chunking {len(documents)} documents ===")
        all_chunks = []
        chunk_id = 0
        for doc in documents:
            chunks = section_aware_chunk(
                text=doc["text"],
                source_file=doc.get("source_file", "unknown"),
                doc_title=doc.get("title", ""),
                url=doc.get("url", ""),
                max_tokens=self.max_chunk_tokens,
                start_chunk_id=chunk_id,
            )
            all_chunks.extend(chunks)
            chunk_id = all_chunks[-1].chunk_id + 1 if all_chunks else 0
            print(f"  {doc.get('source_file', 'doc')}: {len(chunks)} chunks")

        # Re-wire linked list across doc boundaries (within-doc only)
        # Already done per-doc, which is correct — no cross-doc linking

        print(f"\nTotal chunks: {len(all_chunks)}")
        total_tokens = sum(c.token_estimate for c in all_chunks)
        print(f"Total estimated tokens: {total_tokens}")

        # 2. Embed all chunks
        print(f"\n=== Embedding {len(all_chunks)} chunks ===")
        texts = [c.text for c in all_chunks]

        # Batch embed
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # Cosine distance needs normalized vectors
            batch_size=64,
        )
        print(f"Embeddings shape: {embeddings.shape}")

        # 3. Build HNSW index
        print(f"\n=== Building HNSW index (M={self.hnsw_config.m}, efC={self.hnsw_config.ef_construction}) ===")
        index = HnswIndex(self.dim, self.hnsw_config, seed=42)

        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            # Store chunk metadata as JSON string in the HNSW node metadata field
            # This is what VecDB-WASM's get_metadata() returns
            meta = json.dumps({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "section_heading": chunk.section_heading,
                "prev_chunk_id": chunk.prev_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
                "url": chunk.url,
                "doc_title": chunk.doc_title,
            })
            index.insert(embedding.tolist(), metadata=meta)

            if (i + 1) % 500 == 0 or i == len(all_chunks) - 1:
                print(f"  Inserted {i + 1}/{len(all_chunks)}")

        # 4. Compute corpus hash
        corpus_text = "".join(texts)
        corpus_hash = hashlib.sha256(corpus_text.encode()).hexdigest()[:16]

        # 5. Write HNSW index JSON (loadable by VectorDB.load())
        print(f"\n=== Serializing index ===")
        index_path = output / "index.json"
        with open(index_path, "w") as f:
            json.dump(index.to_dict(), f)
        index_size = index_path.stat().st_size
        print(f"Index size: {index_size / (1024*1024):.1f} MB")

        # 6. Write chunks.json (parallel metadata array, useful for clients
        #    that want to browse chunks without parsing HNSW node metadata)
        chunks_path = output / "chunks.json"
        chunks_data = []
        for c in all_chunks:
            chunks_data.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_file": c.source_file,
                "section_heading": c.section_heading,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "token_estimate": c.token_estimate,
                "prev_chunk_id": c.prev_chunk_id,
                "next_chunk_id": c.next_chunk_id,
                "url": c.url,
                "doc_title": c.doc_title,
            })
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2)

        # 7. Write manifest.json
        manifest = {
            "vrom_id": vrom_id,
            "version": version,
            "description": description,
            "source": source,
            "embedding_spec": {
                "model": f"Xenova/{self.model_name}",
                "model_source": f"sentence-transformers/{self.model_name}",
                "dimensions": self.dim,
                "quantization": "q8",
                "distance_metric": "cosine",
                "normalized": True,
            },
            "hnsw_config": self.hnsw_config.to_dict(),
            "vector_count": len(all_chunks),
            "total_tokens": total_tokens,
            "total_chunks": len(all_chunks),
            "corpus_hash": corpus_hash,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunk_strategy": {
                "method": "section_aware",
                "max_tokens": self.max_chunk_tokens,
                "overlap": 0,
                "preserve_code_blocks": True,
                "linked_list_pointers": True,
            },
            "files": {
                "index": "index.json",
                "chunks": "chunks.json",
                "manifest": "manifest.json",
            },
            "compatibility": {
                "vecdb_wasm": ">=0.1.0",
                "load_method": "VectorDB.load(json)",
            },
        }
        manifest_path = output / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n=== vROM package complete ===")
        print(f"  Output: {output}")
        print(f"  Files:")
        print(f"    index.json    : {index_size / (1024*1024):.1f} MB ({len(all_chunks)} vectors)")
        print(f"    chunks.json   : {chunks_path.stat().st_size / 1024:.1f} KB")
        print(f"    manifest.json : {manifest_path.stat().st_size / 1024:.1f} KB")

        return output


# ─── Documentation Fetcher ────────────────────────────────────────────────────

def fetch_hf_docs_pages(
    endpoint: str = "transformers",
    pages: list[str] = None,
) -> list[dict]:
    """
    Fetch markdown documentation pages from HF docs.

    Args:
        endpoint: The HF docs endpoint (transformers, hub, datasets, etc.)
        pages: List of page paths to fetch. If None, fetches a curated set.

    Returns:
        List of {"text": ..., "source_file": ..., "title": ..., "url": ...}
    """
    import requests

    base_url = f"https://huggingface.co/docs/{endpoint}"
    documents = []

    for page in pages:
        url = f"https://huggingface.co/docs/{endpoint}/{page}"
        md_url = f"{url}.md"
        try:
            resp = requests.get(md_url, timeout=30)
            if resp.status_code == 200:
                text = resp.text
                # Extract title from first heading
                title_match = re.match(r'^#\s+(.*?)$', text, re.MULTILINE)
                title = title_match.group(1) if title_match else page

                documents.append({
                    "text": text,
                    "source_file": f"{endpoint}/{page}.md",
                    "title": title,
                    "url": url,
                })
                print(f"  Fetched: {page} ({len(text)} chars)")
            else:
                print(f"  SKIP: {page} (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  ERROR: {page}: {e}")

    return documents


if __name__ == "__main__":
    # Quick test with a small doc
    test_text = """# Getting Started with Transformers

## Installation

You can install transformers with pip:

```bash
pip install transformers
```

This will install the latest stable version.

## Quick Tour

Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.

### Pipeline API

The simplest way to use a pretrained model is with the `pipeline` function:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this library!")
print(result)
```

### AutoModel

For more control, use the AutoModel classes:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
```
"""
    builder = VromBuilder(max_chunk_tokens=200)
    output = builder.build(
        documents=[{
            "text": test_text,
            "source_file": "test/getting_started.md",
            "title": "Getting Started with Transformers",
            "url": "https://example.com/docs/getting-started",
        }],
        vrom_id="test-vrom",
        version="0.0.1",
        description="Test vROM",
        output_dir="/app/test_vrom",
    )

    # Validate: load the index and do a test search
    with open(output / "index.json") as f:
        data = json.load(f)

    print(f"\n=== Validation ===")
    print(f"Index has {len(data['nodes'])} nodes, dim={data['dim']}")
    print(f"Config: M={data['config']['m']}, efC={data['config']['ef_construction']}")

    # Reconstruct and search
    cfg = data['config']
    index = HnswIndex(data['dim'], HnswConfig(
        m=cfg['m'],
        ef_construction=cfg['ef_construction'],
        ef_search=cfg['ef_search'],
        metric=cfg['metric'],
    ))
    index.nodes = [
        HnswNode(n['vector'], n['max_layer'], n['metadata'])
        for n in data['nodes']
    ]
    for i, n in enumerate(data['nodes']):
        index.nodes[i].neighbors = n['neighbors']
    index.entry_point = data['entry_point']
    index.max_layer = data['max_layer']

    # Search for "how to install"
    query_emb = builder.model.encode(["how to install transformers"], normalize_embeddings=True)
    results = index.search(query_emb[0].tolist(), k=3)
    print(f"\nSearch: 'how to install transformers'")
    for dist, nid in results:
        meta = json.loads(index.nodes[nid].metadata)
        print(f"  [{dist:.4f}] {meta['section_heading']}: {meta['text'][:80]}...")
