---
title: VecDB-WASM
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: WASM vector DB + vROM Hub for plug-and-play browser RAG
---

# 🔍 VecDB-WASM — Vector Search Engine + vROM Hub

A **WebAssembly vector search database** with **built-in embedding models**, **persistent storage**, and a **vROM Hub** for distributing pre-computed HNSW indexes. Runs 100% in the browser — zero server dependencies.

## ✨ What's New in v3

- **🧩 vROM Hub** — Browse and load pre-computed HNSW indexes from Hugging Face Hub. One-click load for instant RAG — no embedding computation needed.
- **🔨 vROM Builder** — Create custom vROM packages directly in the browser. Paste text, fetch HF docs, chunk, embed, and package an HNSW index.
- **🔗 Context Expansion** — Linked-list chunk traversal (`prev_chunk_id` / `next_chunk_id`) for expanding search results with surrounding context.
- **🐍 Python CLI** — Build vROMs from large corpora server-side with `tools/vrom_builder.py`.

## What is a vROM?

A **vROM (Vector Read-Only Memory)** is a pre-computed, serialized HNSW index package that can be loaded directly into VecDB-WASM for instant vector search — no embedding computation required on the client side.

Think of it as a **plug-and-play RAG cartridge**: download, load, and search in milliseconds.

### Official vROMs

| vROM | Vectors | Size | Source |
|------|---------|------|--------|
| [vrom-hf-docs](https://huggingface.co/datasets/philipp-zettl/vrom-hf-docs) | 1,356 | 12.6 MB | HF Transformers + Hub docs |
| [vrom-ml-training](https://huggingface.co/datasets/philipp-zettl/vrom-ml-training) | 629 | 5.8 MB | TRL + PEFT + Datasets docs |

## Features

| Feature | Details |
|---------|---------|
| **HNSW Index** | Rust → WASM (172 KB), based on [Malkov & Yashunin 2018](https://arxiv.org/abs/1603.09320) |
| **Embedding Models** | all-MiniLM-L6-v2, bge-small, gte-small via transformers.js |
| **vROM Hub** | Browse and load pre-computed indexes from HF Hub |
| **vROM Builder** | Create custom vROMs in-browser or via Python CLI |
| **Distance Metrics** | Cosine, Euclidean (L2), Dot Product |
| **Persistence** | OPFS auto-save after every mutation, auto-restore on load |
| **Context Expansion** | Linked-list chunk traversal for surrounding context |
| **Export/Import** | Download/upload index as JSON file |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  VecDB-WASM v3 + vROM Architecture                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Text ──→ transformers.js (ONNX) ──→ HNSW Index (Rust/WASM) │
│                                              │                   │
│  💾 OPFS (auto-save/restore, crash-safe)     │                   │
│                                              │                   │
│  🧩 vROM Hub ──→ fetch from HF Hub ─────────┘                   │
│     • Pre-computed HNSW indexes                                  │
│     • One-click load → instant RAG                               │
│     • Context expansion via linked-list chunks                   │
│                                                                  │
│  🔨 vROM Builder                                                 │
│     • Browser: paste/fetch → chunk → embed → build → download    │
│     • Python: tools/vrom_builder.py (for large corpora)          │
│                                                                  │
│  100% client-side · Zero server dependencies · Works offline     │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Using a vROM (easiest)

1. Click **🧩 vROM Hub** tab
2. Click **Load vROM** on any official vROM
3. Wait for download (~5-13 MB)
4. Load the embedding model when prompted
5. Search with natural language queries

### Building from scratch

1. Click **🔍 Engine** tab → **Load Model**
2. Index text via **📝 Text**, **📄 Bulk Text**, or **🤗 HF Datasets**
3. Search with natural language queries

### Building a custom vROM

1. Click **🔨 vROM Builder** tab
2. Add sources (paste text or fetch HF docs pages)
3. Configure (ID, version, chunk size)
4. Click **Build** → Download or Load in Engine

### Python CLI (for large corpora)

```bash
pip install sentence-transformers huggingface_hub
python tools/vrom_builder.py
```

## vROM Package Format

A vROM consists of 3 JSON files:

| File | Description |
|------|-------------|
| `index.json` | HNSW index (loadable by `VectorDB.load()`) |
| `chunks.json` | Chunk metadata array |
| `manifest.json` | Package specification |

Each vector's metadata contains:
```json
{
  "chunk_id": 42,
  "text": "The actual chunk text...",
  "source_file": "transformers/pipeline_tutorial.md",
  "section_heading": "Pipeline API",
  "prev_chunk_id": 41,
  "next_chunk_id": 43,
  "url": "https://huggingface.co/docs/transformers/pipeline_tutorial",
  "doc_title": "Pipeline"
}
```

## Performance

| Metric | Value |
|--------|-------|
| Search Latency | **< 1 ms** (HNSW) |
| Embedding Latency | **~50 ms** per sentence (q8, WASM) |
| WASM Binary | **172 KB** |
| Embedding Model | **22 MB** (q8 all-MiniLM-L6-v2) |
| vROM Load Time | **~2-5s** (depends on size + network) |

## Technology Stack

- **Rust** — HNSW algorithm, distance metrics, serialization
- **wasm-bindgen** — Rust ↔ JavaScript bridge
- **transformers.js** — In-browser ONNX inference for embeddings
- **OPFS** — Browser-native persistent file storage
- **Hugging Face Hub** — vROM distribution via dataset repos
- **Zero frameworks** — Vanilla JS, no build step

## License

MIT
