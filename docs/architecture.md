# Architecture

This document explains how VecDB-WASM works internally — the HNSW algorithm, the WASM engine, the worker protocol, the OPFS cache layer, and the vROM format.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Browser (Main Thread)                                                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  AgentMemory                                                     │   │
│  │  • Lifecycle: init() → mount() → search() → destroy()           │   │
│  │  • Coordinates all subsystems                                    │   │
│  └────────┬─────────────────┬──────────────────────┬────────────────┘   │
│           │                 │                      │                    │
│  ┌────────▼────────┐  ┌────▼─────────────┐  ┌─────▼──────────────┐    │
│  │  VectorDB       │  │  VromCache        │  │  Worker RPC        │    │
│  │  (Rust → WASM)  │  │  (OPFS layer)     │  │  (postMessage)     │    │
│  │                 │  │                   │  │                    │    │
│  │  • HNSW graph   │  │  • Registry cache │  │  ┌──────────────┐ │    │
│  │  • <1ms search  │  │  • vROM cache     │  │  │ Web Worker   │ │    │
│  │  • 172 KB .wasm │  │  • 1h TTL         │  │  │ transformers │ │    │
│  │  • JSON serde   │  │  • Stream DL      │  │  │ .js ONNX     │ │    │
│  └─────────────────┘  └───────────────────┘  │  │ ~50ms/embed  │ │    │
│                                               │  └──────────────┘ │    │
│                                               └───────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  💾 OPFS (Origin Private File System)                           │    │
│  │  vecdb-vroms/registry.json                                      │    │
│  │  vecdb-vroms/{vrom-id}/manifest.json                            │    │
│  │  vecdb-vroms/{vrom-id}/index.json                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
         │                              │
         │  VectorDB.load(json)         │  fetch() from CDN
         │  VectorDB.save()             │
         ▼                              ▼
┌─────────────────┐          ┌──────────────────────┐
│  WASM Memory    │          │  HF Hub CDN          │
│  (linear heap)  │          │  vROM Registry       │
│  vectors +      │          │  Index + Manifest    │
│  HNSW graph     │          │  Model weights (ONNX)│
└─────────────────┘          └──────────────────────┘
```

## The HNSW Algorithm

VecDB-WASM implements **HNSW (Hierarchical Navigable Small World)** for approximate nearest neighbor search. The implementation follows:

- Malkov & Yashunin, *"Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"* (2018), [arxiv:1603.09320](https://arxiv.org/abs/1603.09320)
- Parameter choices from [arxiv:2405.17813](https://arxiv.org/abs/2405.17813)

### How HNSW Works

HNSW builds a multi-layer graph where:

1. **Layer 0** (bottom) contains all vectors with up to `M_max0 = 2M` connections each
2. **Higher layers** contain exponentially fewer vectors, each with up to `M` connections
3. Each vector is randomly assigned to layers 0 through `L`, where `L ~ -ln(random) × level_multiplier`

**Search** starts from the top layer's entry point and greedily descends:

```
Layer 3:  [EP] ──→ nearest in layer 3
            │
Layer 2:    ▼  ──→ nearest in layer 2
            │
Layer 1:    ▼  ──→ nearest in layer 1
            │
Layer 0:    ▼  ──→ beam search with ef candidates → top-k results
```

The top layers act as a "highway system" for fast coarse navigation. Layer 0 does the fine-grained search with a beam width of `efSearch`.

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `M` | 16 | Max connections per node per layer. Higher = better recall, more memory. |
| `M_max0` | 32 | Max connections at layer 0 (always `2 × M`). |
| `efConstruction` | 128 | Beam width during index construction. Higher = better graph quality, slower build. |
| `efSearch` | 40 | Beam width during search. Higher = better recall, slower search. Must be ≥ `k`. |
| `level_multiplier` | `1/ln(M)` | Controls layer assignment distribution. |

### Distance Metrics

All metrics return a value where **lower = more similar**:

| Metric | Formula | Range | Notes |
|--------|---------|-------|-------|
| **Cosine** (default) | `1 - cos(a, b)` | `[0, 2]` | 0 = identical, 1 = orthogonal |
| **Euclidean** | `Σ(aᵢ - bᵢ)²` | `[0, ∞)` | Squared L2 (no sqrt for perf) |
| **Dot Product** | `-Σ(aᵢ × bᵢ)` | `(-∞, ∞)` | Negated so lower = more similar |

The vROM system uses **Cosine** with **normalized vectors**, making cosine distance equivalent to `2 × (1 - dot(a, b))`. This is why all vROM embeddings are L2-normalized during build.

---

## The WASM Engine

The core vector search engine is written in Rust and compiled to WebAssembly.

### Source Structure

```
src/
  Cargo.toml          ← Crate manifest
  rust/
    lib.rs            ← wasm-bindgen API (VectorDB class exposed to JS)
    hnsw.rs           ← HNSW algorithm: insert, search, serialize/deserialize
    distance.rs       ← Cosine, Euclidean, Dot Product distance functions
```

### Compilation

```bash
wasm-pack build src/ --target bundler --out-dir lib/wasm-pkg --release
```

**Release profile** (from `Cargo.toml`):

```toml
[profile.release]
opt-level = "z"      # Optimize for size
lto = true           # Link-time optimization
codegen-units = 1    # Single codegen unit for better optimization
strip = true         # Strip debug info
```

This produces a ~172 KB `.wasm` binary.

### Serialization Format

The HNSW index serializes to JSON via `serde`. The schema is:

```json
{
  "config": {
    "m": 16,
    "m_max0": 32,
    "ef_construction": 128,
    "ef_search": 40,
    "level_multiplier": 0.3607,
    "metric": "Cosine"
  },
  "nodes": [
    {
      "vector": [0.123, -0.456, ...],
      "neighbors": [[12, 45, 78], [3, 67]],
      "max_layer": 1,
      "metadata": "{\"chunk_id\":0,\"text\":\"...\",\"source_file\":\"...\"}"
    }
  ],
  "entry_point": 42,
  "max_layer": 3,
  "dim": 384
}
```

Key design decisions:
- **JSON** (not binary) — for simplicity, debuggability, and web compatibility. The size overhead vs. binary is ~2×, but gzip compression brings it close.
- **Metadata as JSON string** — each node's metadata is a JSON-encoded string. This allows arbitrary metadata without schema changes.
- **Full graph in one file** — no external references. One `VectorDB.load(json)` call hydrates the entire index.

### Memory Model

Each `VectorDB` instance lives in WASM linear memory. The `free()` method deallocates the Rust struct. Calling any method after `free()` will trap (null pointer).

```typescript
const db = VectorDB.load(json);
db.search(query, 5);  // OK
db.free();
db.search(query, 5);  // WASM trap: null pointer
```

The `AgentMemory` class handles this internally — it always calls `free()` before loading a new index.

---

## The Embed Worker

Embedding inference runs in a **Web Worker** to avoid blocking the main thread. The worker loads [transformers.js](https://huggingface.co/docs/transformers.js) from CDN and runs ONNX models.

### Why a Separate Worker?

1. **Non-blocking UI** — ONNX inference takes ~50ms per sentence. Running it on the main thread would cause jank.
2. **Memory isolation** — the ONNX runtime allocates significant memory. The worker can be terminated to reclaim it.
3. **Model singleton** — a single worker instance manages the model lifecycle, preventing duplicate loads.

### Worker Lifecycle

```
                    Main Thread                         Worker Thread
                    ───────────                         ─────────────
                         │                                   │
  new Worker('embed-worker.js', {type:'module'})  ──→       │
                         │                          (spawned, idle)
                         │                                   │
  postMessage({type:'load', modelId, dtype})  ────────→     │
                         │                          pipeline('feature-extraction')
                         │                    ←──── {status:'dl-progress', ...}
                         │                    ←──── {status:'dl-progress', ...}
                         │                          embed(['test']) for dim probe
                         │                    ←──── {status:'ready', dim:384}
                         │                                   │
  postMessage({type:'embed', texts, id})  ─────────→        │
                         │                          extractor(texts, {pooling,norm})
                         │                    ←──── {status:'result', id, embeddings}
                         │                          (Float32Array zero-copy transfer)
```

### Model Diffing

The worker tracks `currentModel` and `currentDtype`. When a `load` message arrives:

- If `modelId === currentModel && dtype === currentDtype` → responds with `{cached: true}` immediately
- Otherwise → disposes the old model, loads the new one

This is what enables hot-swapping between vROMs with zero redundant model loads.

### Zero-Copy Transfer

Embedding results use [`postMessage` with `Transferable`](https://developer.mozilla.org/en-US/docs/Web/API/Worker/postMessage#transfer):

```typescript
const float32 = new Float32Array(output.data);
sendTransfer(
  { status: 'result', id, embeddings: float32, dims: output.dims },
  [float32.buffer],  // Transfer ownership — no copy
);
```

The `Float32Array.buffer` is transferred from the worker's memory to the main thread. This avoids a full copy of the embedding vector.

---

## The OPFS Cache Layer

The `VromCache` class manages persistent storage using the [Origin Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system).

### Why OPFS?

| Storage API | Persistence | Size Limit | Access Speed | Structured Data |
|------------|-------------|------------|--------------|-----------------|
| localStorage | ✓ | 5–10 MB | Sync, fast | Strings only |
| IndexedDB | ✓ | Large | Async, medium | Structured |
| Cache API | ✓ | Large | Async, fast | HTTP responses |
| **OPFS** | ✓ | Large | Async, fast | **Raw files** |

OPFS is ideal because:
- vROM indexes are **raw JSON files** (10–15 MB) — not HTTP responses (Cache API) or structured objects (IndexedDB)
- OPFS has the **lowest overhead** for reading large text files
- The file hierarchy maps naturally to the vROM structure

### Registry Caching

The registry is cached with a **1-hour TTL**:

```typescript
const modTime = await this.#fileModTime(root, 'registry.json');
if (modTime && Date.now() - modTime < 3_600_000) {
  // Use cached registry
} else {
  // Fetch from CDN, write to OPFS
}
```

### Streamed Downloads

Large index files are downloaded with streaming to report progress:

```typescript
const reader = response.body!.getReader();
const chunks: Uint8Array[] = [];
let loaded = 0;

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  chunks.push(value);
  loaded += value.length;
  onProgress?.({ phase: 'index', file: 'index.json', loaded, total });
}
```

---

## vROM Format Specification

### Directory Structure

A vROM package consists of three files:

```
{vrom-id}/
  manifest.json     ← Build metadata and compatibility info
  index.json        ← Serialized HNSW graph (VectorDB.load() compatible)
  chunks.json       ← Parallel chunk metadata array (for CLI/offline tools)
```

### `manifest.json`

```json
{
  "vrom_id": "hf-transformers-docs",
  "version": "1.0.0",
  "description": "HuggingFace Transformers documentation",
  "source": "https://huggingface.co/docs/transformers",
  "embedding_spec": {
    "model": "Xenova/all-MiniLM-L6-v2",
    "model_source": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "quantization": "q8",
    "distance_metric": "cosine",
    "normalized": true
  },
  "hnsw_config": {
    "m": 16,
    "m_max0": 32,
    "ef_construction": 128,
    "ef_search": 40,
    "level_multiplier": 0.3607,
    "metric": "Cosine"
  },
  "vector_count": 1356,
  "total_tokens": 250000,
  "total_chunks": 1356,
  "corpus_hash": "a1b2c3d4e5f6g7h8",
  "created_at": "2025-01-15T12:00:00Z",
  "chunk_strategy": {
    "method": "section_aware",
    "max_tokens": 256,
    "overlap": 0,
    "preserve_code_blocks": true,
    "linked_list_pointers": true
  },
  "compatibility": {
    "vecdb_wasm": ">=0.1.0",
    "load_method": "VectorDB.load(json)"
  }
}
```

### `index.json`

The HNSW index in the exact `serde` JSON format that `VectorDB.load()` expects (see [Serialization Format](#serialization-format) above).

### `chunks.json`

```json
[
  {
    "chunk_id": 0,
    "text": "# Quick Tour\n\nTransformers provides...",
    "source_file": "transformers/quicktour.md",
    "section_heading": "Quick Tour",
    "char_start": 0,
    "char_end": 512,
    "token_estimate": 128,
    "prev_chunk_id": null,
    "next_chunk_id": 1,
    "url": "https://huggingface.co/docs/transformers/quicktour",
    "doc_title": "Quick Tour"
  }
]
```

This file is **not** loaded by the browser SDK — it's provided for the Python CLI (`vrom_cli.py search`) and offline inspection tools.

---

## Build Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Rust Source      │     │  TypeScript Src   │     │  npm Package     │
│  src/rust/*.rs    │     │  lib/src/*.ts     │     │  lib/dist/       │
│                   │     │                   │     │                  │
│  distance.rs ─┐   │     │  index.ts ─────┐  │     │  index.js (ESM)  │
│  hnsw.rs ─────┤   │────▶│  agent-memory ─┤  │────▶│  index.cjs(CJS)  │
│  lib.rs ──────┘   │     │  vrom-cache ───┤  │     │  index.d.ts      │
│               wasm│     │  embed-worker ─┤  │     │  embed-worker.js │
│               pack│     │  types.ts ─────┘  │     │  embed-worker.d  │
│                   │     │              tsdown│     │  vecdb_wasm.wasm │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                         │
         ▼                         ▼
  lib/wasm-pkg/            lib/dist/
  vecdb_wasm.js            (copied from wasm-pkg)
  vecdb_wasm.d.ts
  vecdb_wasm_bg.wasm
```

### Stage 1: Rust → WASM

`wasm-pack build` compiles the Rust crate with:
- `wasm-bindgen` for the JS bridge
- `serde` + `serde_json` for JSON serialization
- `console_error_panic_hook` for debuggable WASM panics
- `rand` with `SmallRng` for HNSW level assignment

### Stage 2: TypeScript → npm

`tsdown` bundles from `lib/src/`:
1. **Main entry** (`index.ts`) → ESM + CJS + `.d.ts` declarations
2. **Worker entry** (`embed-worker.ts`) → separate ESM file + `.d.ts`
3. **Post-build** copies `.wasm` binary and stabilizes `.d.ts` filenames

---

## Design Decisions

### Why WASM for Vector Search?

JavaScript is too slow for distance computations on thousands of vectors. The Rust HNSW implementation in WASM is ~10–50× faster than equivalent JS, achieving sub-millisecond search on 1000+ vectors.

### Why JSON Serialization?

Binary formats (protobuf, FlatBuffers) would be smaller and faster to parse, but JSON provides:
- Zero-dependency parsing in both Rust (`serde_json`) and JS (`JSON.parse`)
- Human-readable debugging
- gzip compression on CDN closes the size gap (~3× compression ratio)

The tradeoff is a ~200ms parse time for a 12 MB index, which is acceptable for a one-time load.

### Why transformers.js from CDN?

The embed worker imports transformers.js from `https://cdn.jsdelivr.net/npm/@huggingface/transformers@3`. This avoids bundling the ~2 MB transformers.js library into the npm package. The CDN version is cached by the browser.

### Why OPFS Instead of IndexedDB?

OPFS handles large (10+ MB) text files more efficiently than IndexedDB, with lower overhead and a simpler API for file-based access patterns. IndexedDB requires transaction management and structured cloning, which adds latency for large blobs.

### Why Zero Chunk Overlap?

Following research (arxiv:2601.14123), the chunking strategy uses zero overlap between chunks. The linked-list pointer system (`prev_chunk_id`/`next_chunk_id`) provides on-demand context expansion at search time, which is more flexible and storage-efficient than pre-computed overlapping chunks.
