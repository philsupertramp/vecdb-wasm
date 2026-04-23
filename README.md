---
title: VecDB-WASM
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: WASM vector DB with HNSW, embeddings, OPFS
---

# 🔍 VecDB-WASM — Vector Search Database Engine

A **WebAssembly vector search database** with **built-in embedding models** and **persistent storage**. Runs 100% in the browser — zero server dependencies.

## ✨ What's New in v2

- **🧠 Embedding Integration** — Load transformer models directly in the browser via [transformers.js](https://huggingface.co/docs/transformers.js). Index and search with raw text.
- **💾 OPFS Persistence** — Your data survives page refreshes. Uses the [Origin Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) with atomic writes.
- **📝 Text-First Workflow** — Type text → auto-embed → index. Search with natural language queries.

## Features

| Feature | Details |
|---------|---------|
| **HNSW Index** | Rust → WASM (172 KB), based on [Malkov & Yashunin 2018](https://arxiv.org/abs/1603.09320) |
| **Embedding Models** | all-MiniLM-L6-v2, bge-small, gte-small via transformers.js |
| **Distance Metrics** | Cosine, Euclidean (L2), Dot Product |
| **Persistence** | OPFS auto-save after every mutation, auto-restore on load |
| **Quantization** | q8 (22MB) or fp32 embedding models |
| **Batch Operations** | Bulk text indexing (32 lines/batch), batch vector insert |
| **Export/Import** | Download/upload index as JSON file |

## How It Works

```
Raw Text → Embedding Model (transformers.js, in-browser ONNX)
         → Float32Array (384-dim vector)
         → HNSW Index (Rust/WASM, sub-ms search)
         → OPFS (auto-persisted, survives refresh)
```

## Quick Start

1. **Load Model** — Click "Load Model" to download an embedding model (~22MB, cached after first load)
2. **Index Text** — Type or paste text in the "Index Data" panel
3. **Search** — Type a natural language query in "Text Search"
4. **Refresh** — Your data is still there (OPFS persistence)

## Performance

| Metric | Value |
|--------|-------|
| Search Latency | **< 1 ms** (HNSW) |
| Embedding Latency | **~50 ms** per sentence (q8, WASM) |
| Recall@10 | **100%** (500 vectors) |
| WASM Binary | **172 KB** |
| Embedding Model | **22 MB** (q8 all-MiniLM-L6-v2) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Raw Text ──→ transformers.js (ONNX) ──→ HNSW Index (Rust/WASM)│
│                                              │                  │
│              OPFS (Origin Private File System)                  │
│              • Auto-save after mutations                        │
│              • Auto-restore on page load                        │
│              • Atomic writes (crash-safe)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Rust** — HNSW algorithm, distance metrics, serialization
- **wasm-bindgen** — Rust ↔ JavaScript bridge
- **transformers.js** — In-browser ONNX inference for embeddings
- **OPFS** — Browser-native persistent file storage
- **Zero frameworks** — Vanilla JS, no build step needed

## Browser Compatibility

| Browser | WASM | OPFS | Embeddings |
|---------|------|------|------------|
| Chrome 102+ | ✅ | ✅ | ✅ |
| Firefox 111+ | ✅ | ✅ | ✅ |
| Safari 15.2+ | ✅ | ✅ | ✅ |
| Edge | ✅ | ✅ | ✅ |

## License

MIT
