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

A **WebAssembly vector search database** with **background embedding**, **persistent storage**, and a **vROM Hub** for distributing pre-computed HNSW indexes. Runs 100% in the browser — zero server dependencies.

## ✨ What's New in v3.1

- **🧵 Background Embedding Worker** — ONNX inference runs in a Web Worker, keeping the UI completely responsive during embedding. Zero main-thread blocking.
- **📡 CDN Registry** — vROMs served from a [centralized registry](https://huggingface.co/datasets/philipp-zettl/vrom-registry) via HF Hub CDN (`resolve/main/` URLs, Cloudflare-cached, ETag support).
- **🐍 vROM CLI** — Command-line tool for managing vROMs locally: `vrom list`, `vrom pull`, `vrom search`.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  VecDB-WASM v3.1 Architecture                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Main Thread                    Web Worker Thread                │
│  ┌──────────────┐              ┌──────────────────┐              │
│  │ UI + HNSW    │  postMessage │ transformers.js   │              │
│  │ (Rust/WASM)  │◄────────────►│ ONNX embedding    │              │
│  │ VectorDB     │  Float32Array│ all-MiniLM-L6-v2  │              │
│  │ search <1ms  │  (zero-copy) │ q8 quantized      │              │
│  └──────────────┘              └──────────────────┘              │
│         │                                                        │
│  ┌──────┴───────────────────────────────────────────────────┐    │
│  │  💾 OPFS — auto-save/restore · crash-safe                │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  🧩 vROM Hub — CDN-distributed pre-computed indexes      │    │
│  │  registry.json → fetch index.json → VectorDB.load()      │    │
│  │  Context expansion via prev/next chunk linked-list        │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  100% client-side · Works offline · Non-blocking UI              │
└──────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Details |
|---------|---------|
| **HNSW Index** | Rust → WASM (172 KB), [Malkov & Yashunin 2018](https://arxiv.org/abs/1603.09320) |
| **Background Embedding** | Web Worker thread — UI never blocks during inference |
| **vROM Hub** | Browse and load pre-computed indexes from CDN |
| **vROM Builder** | Create custom vROMs in-browser or via Python CLI |
| **vROM CLI** | `vrom list` / `pull` / `search` — local management |
| **Distance Metrics** | Cosine, Euclidean (L2), Dot Product |
| **Persistence** | OPFS auto-save/restore, crash-safe |
| **Context Expansion** | Linked-list chunk traversal |

## Official vROMs

Served from the [vROM Registry](https://huggingface.co/datasets/philipp-zettl/vrom-registry):

| vROM | Vectors | Size | Source |
|------|---------|------|--------|
| `hf-transformers-docs` | 1,356 | 12.6 MB | HF Transformers + Hub docs |
| `hf-ml-training` | 629 | 5.8 MB | TRL + PEFT + Datasets docs |

## vROM CLI

```bash
# Install (just needs requests; sentence-transformers for search)
pip install requests sentence-transformers

# List available vROMs
python tools/vrom_cli.py list

# Download a vROM to ~/.vrom/
python tools/vrom_cli.py pull hf-transformers-docs

# Search locally
python tools/vrom_cli.py search hf-ml-training "how to train with DPO"

# Show details
python tools/vrom_cli.py info hf-transformers-docs
```

## File Structure

```
index.html              # Main application (CSS + HTML)
embed-worker.js         # Web Worker — transformers.js ONNX inference
pkg/                    # VecDB-WASM compiled binaries
  vecdb_wasm.js         # JS bindings
  vecdb_wasm_bg.wasm    # WASM binary (172 KB)
src/rust/               # Rust source code
  hnsw.rs               # HNSW algorithm
  distance.rs           # Distance metrics
  lib.rs                # wasm-bindgen API
tools/
  vrom_builder.py       # Python vROM builder (for large corpora)
  vrom_cli.py           # CLI for local vROM management
```

## Performance

| Metric | Value |
|--------|-------|
| HNSW Search | **< 1 ms** |
| Embedding (worker) | **~50 ms** per sentence |
| UI blocking during embed | **0 ms** (worker thread) |
| WASM Binary | **172 KB** |
| Embedding Model | **22 MB** (q8) |

## License

MIT
