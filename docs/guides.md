# Guides

## Table of Contents

- [Understanding vROMs](#understanding-vroms)
- [Context Expansion](#context-expansion)
- [Building Custom vROMs](#building-custom-vroms)
- [Cache Management](#cache-management)
- [Model Diffing and Hot-Swapping](#model-diffing-and-hot-swapping)
- [Performance Tuning](#performance-tuning)
- [Error Handling](#error-handling)
- [Python CLI](#python-cli)

---

## Understanding vROMs

A **vROM** (Vector Read-Only Memory) is a pre-computed knowledge base package for VecDB-WASM. Think of it like a ROM cartridge: you slot it in, and the knowledge is instantly available for search.

### What's Inside a vROM

Each vROM contains three files:

| File | Contents | Size |
|------|----------|------|
| `manifest.json` | Metadata: model requirements, HNSW config, corpus hash, version | ~1 KB |
| `index.json` | Serialized HNSW graph with all vectors + chunk metadata | 5–15 MB |
| `chunks.json` | Parallel array of chunk metadata (for offline browsing tools) | 200 KB–2 MB |

The `index.json` is the critical file — it contains the complete HNSW graph that `VectorDB.load()` deserializes directly. This includes:
- All vector embeddings (Float32)
- Per-node HNSW neighbor lists across all layers
- Per-vector metadata (chunk text, source file, heading, URL, linked-list pointers)
- Graph configuration (M, efConstruction, metric)

### Official vROMs

These are published to the [vROM Registry](https://huggingface.co/datasets/philipp-zettl/vrom-registry):

| ID | Vectors | Download | Content |
|----|---------|----------|---------|
| `hf-transformers-docs` | 1,356 | 12.6 MB | HF Transformers documentation + Hub docs |
| `hf-ml-training` | 629 | 5.8 MB | TRL, PEFT, and Datasets documentation |

Both use `Xenova/all-MiniLM-L6-v2` (384 dimensions, q8 quantized) for embeddings.

### The vROM Registry

The registry is a JSON file hosted on HF Hub that lists all available vROMs with their CDN URLs:

```json
{
  "version": "1.0",
  "description": "VecDB-WASM vROM Registry",
  "base_url": "https://huggingface.co/datasets/...",
  "vroms": [
    {
      "id": "hf-transformers-docs",
      "name": "HF Transformers Docs",
      "description": "HuggingFace Transformers + Hub documentation",
      "version": "1.0.0",
      "vectors": 1356,
      "dimensions": 384,
      "tokens": 250000,
      "size_mb": 12.6,
      "model": "Xenova/all-MiniLM-L6-v2",
      "tags": ["huggingface", "transformers", "documentation"],
      "official": true,
      "files": {
        "manifest": "https://huggingface.co/.../manifest.json",
        "index": "https://huggingface.co/.../index.json",
        "chunks": "https://huggingface.co/.../chunks.json"
      }
    }
  ]
}
```

You can point to a custom registry by passing `registryUrl` to the `AgentMemory` constructor:

```typescript
const memory = new AgentMemory({
  registryUrl: 'https://my-cdn.com/my-registry.json',
});
```

---

## Context Expansion

VecDB-WASM chunks documents into ~256-token segments. Each chunk stores linked-list pointers (`prev_chunk_id`, `next_chunk_id`) to its neighbors in the original document. Context expansion follows these pointers to reassemble larger context windows.

### Why Context Expansion Matters

A single 256-token chunk may not contain enough context for an LLM to understand the full answer. Context expansion retrieves the chunks before and after each search result, reconstructing the original document flow.

### How It Works

```
Document:  [chunk 0] ←→ [chunk 1] ←→ [chunk 2] ←→ [chunk 3] ←→ [chunk 4]

Search hit: chunk 2 (distance: 0.15)

expandContext: false → returns just chunk 2
expandContext: true, contextWindow: 1 → returns [chunk 1, chunk 2, chunk 3]
expandContext: true, contextWindow: 2 → returns [chunk 0, chunk 1, chunk 2, chunk 3, chunk 4]
```

The expanded text is joined with `\n\n` separators. The metadata is annotated:
- `result.metadata._expanded = true`
- `result.metadata._contextChunks = 5` (total chunks in the expansion)

### Usage

```typescript
const results = await memory.search('how to create a pipeline', {
  topK: 3,
  expandContext: true,
  contextWindow: 1,  // 1 chunk before + 1 after = 3 chunks per result
});

// Each result now has ~768 tokens of context instead of ~256
const context = memory.formatContext(results, { maxTokens: 3000 });
```

### Recommendations

| Use Case | `expandContext` | `contextWindow` | `topK` |
|----------|----------------|-----------------|--------|
| Quick factoid lookup | `false` | — | 3–5 |
| LLM context injection | `true` | 1 | 3–5 |
| Deep research | `true` | 2 | 5–10 |
| Maximum recall | `true` | 3 | 10 |

> **Note:** Cross-document boundaries are respected. If a chunk is the first in its document, `prev_chunk_id` is `null` and expansion stops there.

---

## Building Custom vROMs

You can create vROMs from your own documentation using either the **browser-based builder** (in the Space UI) or the **Python builder** (for larger corpora).

### Browser Builder (Small Corpora)

The Space UI at [vecdb-wasm](https://huggingface.co/spaces/philipp-zettl/vecdb-wasm) includes a **vROM Builder** tab:

1. Paste markdown text or fetch an HF docs page
2. Configure: vROM ID, version, max chunk tokens
3. Click "Build vROM" — chunks, embeds, and builds the HNSW index in-browser
4. Download the package or load it directly in the Engine tab

This works for documents up to ~500 chunks. For larger corpora, use the Python builder.

### Python Builder (Large Corpora)

The `tools/vrom_builder.py` script builds vROMs from any text source.

**Prerequisites:**

```bash
pip install sentence-transformers huggingface_hub numpy
```

**Basic Usage:**

```python
from vrom_builder import VromBuilder

builder = VromBuilder(
    model_name='all-MiniLM-L6-v2',
    dim=384,
    hnsw_m=16,
    hnsw_ef_construction=128,
    max_chunk_tokens=256,
)

documents = [
    {
        'text': open('docs/getting-started.md').read(),
        'source_file': 'getting-started.md',
        'title': 'Getting Started',
        'url': 'https://example.com/docs/getting-started',
    },
    {
        'text': open('docs/api-reference.md').read(),
        'source_file': 'api-reference.md',
        'title': 'API Reference',
        'url': 'https://example.com/docs/api-reference',
    },
]

output = builder.build(
    documents=documents,
    vrom_id='my-docs',
    version='1.0.0',
    description='My project documentation',
    output_dir='./my-vrom',
)
# Creates: my-vrom/index.json, my-vrom/chunks.json, my-vrom/manifest.json
```

**Fetching HF Docs:**

```python
from vrom_builder import VromBuilder, fetch_hf_docs_pages

# Fetch specific pages from HF documentation
documents = fetch_hf_docs_pages(
    endpoint='transformers',
    pages=['quicktour', 'pipeline_tutorial', 'training', 'tokenizer_summary'],
)

builder = VromBuilder()
builder.build(
    documents=documents,
    vrom_id='transformers-quickstart',
    version='1.0.0',
    description='Transformers quick start guides',
    output_dir='./transformers-vrom',
)
```

### Chunking Strategy

The builder uses **section-aware chunking**:

1. Split on markdown headings (`#`, `##`, `###`, etc.)
2. Keep code blocks intact within chunks (never split a fenced code block)
3. Target ~256 tokens per chunk (configurable via `max_chunk_tokens`)
4. **Zero overlap** — research (arxiv:2601.14123) shows overlap adds cost without improving retrieval
5. Wire linked-list pointers (`prev_chunk_id`/`next_chunk_id`) for context expansion

### Publishing a vROM

To make your vROM available via the registry, upload the three files to a HF Hub dataset repo and add an entry to the registry JSON. See the [vROM Registry](https://huggingface.co/datasets/philipp-zettl/vrom-registry) for the schema.

---

## Cache Management

VecDB-WASM uses [OPFS (Origin Private File System)](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) for persistent caching. OPFS is a browser-native filesystem that's:

- **Persistent** across sessions (survives tab close, browser restart)
- **Origin-scoped** (isolated per domain)
- **Not user-visible** (unlike localStorage or downloads)

### Storage Layout

```
[OPFS root]/
  vecdb-vroms/
    registry.json              ← Cached registry (1-hour TTL)
    hf-transformers-docs/
      manifest.json
      index.json               ← The large file (~12 MB)
    hf-ml-training/
      manifest.json
      index.json
```

### Managing the Cache

```typescript
// Check what's cached
const vroms = await memory.listVroms();
for (const v of vroms) {
  const cached = await memory.isCached(v.id);
  console.log(`${v.id}: ${cached ? 'cached' : 'not cached'}`);
}

// Check storage usage
const { used, quota } = await memory.storageEstimate();
console.log(`Using ${(used / 1e6).toFixed(1)} MB of ${(quota / 1e6).toFixed(0)} MB`);

// Evict a specific vROM
await memory.evict('hf-transformers-docs');

// Force re-download (ignore cache)
await memory.mount('hf-transformers-docs', { forceDownload: true });
```

### Cache Behavior

| Operation | Cache Hit | Cache Miss |
|-----------|-----------|------------|
| `mount(id)` | Load from OPFS (~100ms) | Download from CDN, save to OPFS, then load |
| `mount(id, { forceDownload: true })` | Ignored — always re-downloads | Download from CDN |
| `unmount()` | Cache preserved | — |
| `evict(id)` | Files deleted | No-op |
| `destroy()` | Cache preserved | — |

The **embedding model** is cached separately by the browser's Cache API (managed by transformers.js), not by OPFS. Model caching is automatic and persists across sessions.

---

## Model Diffing and Hot-Swapping

When you switch between vROMs, VecDB-WASM automatically manages the embedding model:

```typescript
// First mount — loads all-MiniLM-L6-v2
await memory.mount('hf-transformers-docs');  // Downloads model (~22 MB)

// Second mount — same model required → instant swap
await memory.mount('hf-ml-training');  // Model already loaded, skips reload!
```

This is called **model diffing**. On each `mount()`, the SDK:

1. Reads the `embedding_spec.model` and `embedding_spec.quantization` from the vROM manifest
2. Compares with the currently loaded model ID and dtype
3. If they match → skips model reload (saves 5–10 seconds)
4. If they differ → loads the new model in the background worker

This means switching between vROMs that use the same model is nearly instant (~100ms for OPFS load + WASM deserialization).

---

## Performance Tuning

### HNSW Parameters

The HNSW `efSearch` parameter controls the quality/speed tradeoff at search time:

```typescript
// Default efSearch (typically 40) — fast, good quality
const results = await memory.search('query', { topK: 5 });

// Higher efSearch — slower but better recall
const results = await memory.search('query', { topK: 5, efSearch: 100 });

// Must be >= topK
const results = await memory.search('query', { topK: 20, efSearch: 100 });
```

| efSearch | Latency | Recall | Use Case |
|----------|---------|--------|----------|
| 10–20 | < 0.5 ms | ~90% | Real-time autocomplete |
| 40 (default) | < 1 ms | ~95% | General search |
| 100–200 | 1–3 ms | ~99% | High-precision retrieval |
| 500+ | 5–10 ms | ~100% | Benchmark / evaluation |

### Token Budget Management

Use `formatContext()` with `maxTokens` to fit results within your LLM's context window:

```typescript
// GPT-4o with 128K context — be generous
const context = memory.formatContext(results, { maxTokens: 4000 });

// Small local model with 2K context — be strict
const context = memory.formatContext(results, { maxTokens: 500 });
```

Token estimation uses `ceil(text.length / 4)` — a rough heuristic for English text. For precise budgeting, count tokens with your model's tokenizer after formatting.

### Preloading

If you know which vROM the user will need, preload it during idle time:

```typescript
// During page load or user login
const memory = new AgentMemory();
await memory.init();

// Preload in background — OPFS cache means this is a one-time cost
await memory.mount('hf-transformers-docs');

// Later, when user searches, it's instant
const results = await memory.search(userQuery);
```

---

## Error Handling

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `'Call init() first'` | Called `mount()`/`search()` before `init()` | Call `await memory.init()` first |
| `'vROM \'...\' not found in registry'` | Invalid vROM ID | Check `await memory.listVroms()` for valid IDs |
| `'No vROM mounted — call mount() first'` | Called `search()` without mounting | Call `await memory.mount(id)` first |
| `'Embedding model not loaded'` | Model failed to load or was unloaded | Check network/worker errors; re-mount |
| `'No embedding model loaded'` | Internal embed called before model ready | Ensure `mount()` completes before searching |

### Robust Usage Pattern

```typescript
const memory = new AgentMemory({ logLevel: 'info' });

try {
  await memory.init();
} catch (e) {
  console.error('WASM or Worker init failed:', e);
  // Fallback: WASM not supported, CSP blocks workers, etc.
  return;
}

try {
  await memory.mount('hf-transformers-docs');
} catch (e) {
  console.error('Mount failed:', e);
  // Likely: network error, registry down, OPFS full
  return;
}

// Guard searches with isReady
if (memory.isReady) {
  try {
    const results = await memory.search(query);
    // ... use results
  } catch (e) {
    console.error('Search failed:', e);
  }
}
```

### Logging

Enable verbose logging during development:

```typescript
const memory = new AgentMemory({ logLevel: 'debug' });
```

Log levels output to the browser console with `[AgentMemory:<level>]` prefixes:

| Level | Output |
|-------|--------|
| `silent` | Nothing |
| `error` | Errors only |
| `warn` | Errors + warnings (default) |
| `info` | Lifecycle events (init, mount, model load, search) |
| `debug` | Everything including per-file download progress |

---

## Python CLI

The `tools/vrom_cli.py` script provides command-line access to the vROM ecosystem.

### Installation

```bash
pip install requests
# For search functionality:
pip install sentence-transformers numpy
```

### Commands

#### `list` — List Available vROMs

```bash
python tools/vrom_cli.py list
```

```
ID                             Vectors     Size  Model                     Tags
────────────────────────────────────────────────────────────────────────────────────────────────
★ hf-transformers-docs            1356   12.6MB  Xenova/all-MiniLM-L6-v2   huggingface, transformers
★ hf-ml-training                   629    5.8MB  Xenova/all-MiniLM-L6-v2   training, trl, peft

Local cache: ~/.vrom
  ✓ hf-transformers-docs (12.6 MB)
```

#### `pull` — Download a vROM

```bash
# Download a specific vROM
python tools/vrom_cli.py pull hf-transformers-docs

# Download all vROMs
python tools/vrom_cli.py pull --all

# Force re-download
python tools/vrom_cli.py pull hf-transformers-docs --force
```

Files are saved to `~/.vrom/vroms/<vrom-id>/`.

#### `info` — Show vROM Details

```bash
python tools/vrom_cli.py info hf-transformers-docs
```

```
🧩 HF Transformers Docs
   ID:          hf-transformers-docs
   Version:     1.0.0
   Vectors:     1,356
   Dimensions:  384
   Size:        12.6 MB
   Model:       Xenova/all-MiniLM-L6-v2
```

#### `search` — Search a Local vROM

```bash
python tools/vrom_cli.py search hf-transformers-docs "how to use pipelines"
python tools/vrom_cli.py search hf-ml-training "DPO training" -k 10
```

> **Note:** The Python CLI uses brute-force search (not HNSW) since it doesn't include the Rust engine. Results are identical but slower for large indexes.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VROM_HOME` | `~/.vrom` | Local cache directory |
