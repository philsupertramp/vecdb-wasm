---
title: VecDB-WASM
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: WASM vector DB + AgentMemory SDK for browser RAG
---

# 🔍 VecDB-WASM — Vector Search Engine + AgentMemory SDK

A **WebAssembly vector search database** with a **zero-boilerplate SDK** for browser-based RAG. HNSW search in <1ms, background ONNX embedding, OPFS-cached vROM cartridges, and context hot-swapping — all 100% client-side.

## ✨ What's New in v3.2

- **🧠 AgentMemory SDK** — Single-class API: `init()` → `mount('hub://vrom-name')` → `search("query")`. Handles WASM init, worker spawning, OPFS caching, CDN fetching, and model diffing automatically.
- **🔄 Context Hot-Swapping** — Switch domain expertise on the fly. `mount()` flushes the HNSW graph, resolves from OPFS cache (offline-first), and diffs the embedding model to skip unnecessary reloads.
- **📝 LLM-Ready Output** — `formatContext()` produces structured context strings ready for any LLM prompt window.

## SDK Quick Start

```javascript
import { AgentMemory } from './sdk/vecdb-sdk.js';

const memory = new AgentMemory({ logLevel: 'info' });
await memory.init();

// Mount a vROM — auto-caches to OPFS, auto-loads embedding model
await memory.mount('hf-transformers-docs');

// Search with context expansion (linked-list traversal)
const results = await memory.search("how to use pipelines", {
    topK: 3,
    expandContext: true,
});

// Format for LLM injection
const context = memory.formatContext(results, { maxTokens: 2000 });

// Hot-swap to different domain — model stays loaded if compatible
await memory.mount('hf-ml-training');
```

[**→ Try the interactive SDK demo**](https://huggingface.co/spaces/philipp-zettl/vecdb-wasm/blob/main/sdk/demo.html)

## SDK API Reference

### `new AgentMemory(options?)`
| Option | Default | Description |
|--------|---------|-------------|
| `workerPath` | `'./sdk/embed-worker.js'` | Path to embed worker |
| `wasmPkgPath` | `'./pkg/vecdb_wasm.js'` | Path to WASM bindings |
| `registryUrl` | CDN registry | Custom vROM registry URL |
| `logLevel` | `'warn'` | `silent\|error\|warn\|info\|debug` |

### Methods
| Method | Returns | Description |
|--------|---------|-------------|
| `init()` | `Promise<void>` | Initialize WASM + spawn worker |
| `mount(id, opts?)` | `Promise<MountStatus>` | Mount a vROM (OPFS cache → CDN fallback → model diff) |
| `unmount()` | `void` | Free HNSW graph from RAM |
| `search(query, opts?)` | `Promise<SearchResult[]>` | Embed + HNSW search + optional context expansion |
| `formatContext(results, opts?)` | `string` | Format results as LLM context string |
| `getMountStatus()` | `MountStatus` | Current state: vROM, vectors, model, etc. |
| `listVroms()` | `Promise<object[]>` | List all available vROMs from registry |
| `isCached(id)` | `Promise<boolean>` | Check OPFS cache |
| `evict(id)` | `Promise<void>` | Remove vROM from OPFS cache |
| `destroy()` | `void` | Free all resources + terminate worker |

### Search Options
```typescript
{ topK: 5, expandContext: true, contextWindow: 2, efSearch: 100 }
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  AgentMemory SDK                                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  mount() → resolve hub:// → OPFS cache → VectorDB.load()    │   │
│  │  search() → embed(worker) → HNSW search → expand context    │   │
│  │  formatContext() → LLM-ready string                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Main Thread                     Web Worker Thread                  │
│  ┌──────────────┐               ┌──────────────────┐               │
│  │ VectorDB     │  postMessage  │ transformers.js   │               │
│  │ (Rust/WASM)  │◄─────────────►│ ONNX embedding    │               │
│  │ HNSW <1ms    │  Float32Array │ model diffing     │               │
│  └──────────────┘  (zero-copy)  └──────────────────┘               │
│         │                                                           │
│  ┌──────┴────────────────────────────────────────────────────┐     │
│  │  💾 OPFS — vROM cache (offline-first) + index persistence  │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  📡 vROM Registry CDN — HF Hub resolve/main/ endpoints     │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  100% client-side · Offline-capable · Non-blocking UI               │
└─────────────────────────────────────────────────────────────────────┘
```

## Hot-Swap Lifecycle

When `mount()` is called:

1. **Resolve** — `hub://vrom-id` → registry lookup → CDN URLs
2. **Cache Check** — OPFS hit? Load from disk (sub-second). Miss? Stream from CDN → cache to OPFS.
3. **WASM Flush** — `db.free()` drops old HNSW graph from RAM.
4. **Load** — `VectorDB.load(json)` restores the new graph.
5. **Model Diff** — Compare `embedding_spec.model` of new vROM vs loaded ONNX model. Same model? Skip reload. Different? Hot-swap in the worker thread silently.

## Official vROMs

| ID | Vectors | Size | Source |
|----|---------|------|--------|
| `hf-transformers-docs` | 1,356 | 12.6 MB | HF Transformers + Hub |
| `hf-ml-training` | 629 | 5.8 MB | TRL + PEFT + Datasets |

Registry: [philipp-zettl/vrom-registry](https://huggingface.co/datasets/philipp-zettl/vrom-registry)

## File Structure

```
sdk/                       # AgentMemory SDK
  vecdb-sdk.js             #   AgentMemory class
  vrom-cache.js            #   OPFS cache + hub:// resolver
  embed-worker.js          #   Background ONNX worker (model diffing)
  demo.html                #   Interactive demo page
index.html                 # Main Space UI (Engine + vROM Hub + Builder)
embed-worker.js            # Worker for the main Space UI
pkg/                       # WASM binaries
  vecdb_wasm.js            #   JS bindings
  vecdb_wasm_bg.wasm       #   WASM binary (172 KB)
src/rust/                  # Rust source
  hnsw.rs / distance.rs / lib.rs
tools/
  vrom_builder.py          # Python vROM builder
  vrom_cli.py              # CLI for local vROM management
```

## vROM CLI

```bash
pip install requests sentence-transformers
python tools/vrom_cli.py list
python tools/vrom_cli.py pull hf-transformers-docs
python tools/vrom_cli.py search hf-ml-training "DPO training"
```

## Performance

| Metric | Value |
|--------|-------|
| HNSW Search | **< 1 ms** |
| Embedding (worker) | **~50 ms**/sentence |
| vROM mount (cached) | **< 500 ms** |
| vROM hot-swap (same model) | **< 500 ms** |
| WASM Binary | **172 KB** |

## License

MIT
