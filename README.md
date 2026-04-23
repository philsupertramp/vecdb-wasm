---
title: VecDB-WASM
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
short_description: WASM vector search database with HNSW indexing
---

# 🔍 VecDB-WASM — Vector Search Database Engine

A **WebAssembly vector search database** built from scratch in Rust. Runs entirely in the browser with zero server dependencies.

## Features

- **HNSW Index** — Hierarchical Navigable Small World graph for approximate nearest neighbor search (based on [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320))
- **3 Distance Metrics** — Cosine, Euclidean (L2), Dot Product
- **172 KB** WASM binary — fast to load, no CDN needed
- **Persistence** — Save/load index to JSON (compatible with IndexedDB, localStorage)
- **Configurable** — Tune M, efConstruction, efSearch parameters
- **Metadata Support** — Attach JSON metadata to each vector
- **Batch Insert** — Efficient bulk loading via flat Float32Array

## Performance (Node.js, 10K vectors × 128 dims)

| Metric | Value |
|--------|-------|
| Search Latency | **0.54 ms/query** |
| Throughput | **1,852 QPS** |
| Recall@10 | **100%** (500 vectors) |
| Memory | **6.5 MB** |
| WASM Binary | **172 KB** |

## JavaScript API

```javascript
import init, { VectorDB } from './pkg/vecdb_wasm.js';
await init();

// Create index
const db = new VectorDB(128, 'cosine', 16, 128, 40);

// Insert vectors
db.insert(new Float32Array([0.1, 0.2, ...]), '{"label": "doc1"}');

// Batch insert (flat array)
const vectors = new Float32Array(1000 * 128);
db.insert_batch(vectors, 1000);

// Search
const results = JSON.parse(db.search(queryVector, 10));
// → [{"id": 42, "distance": 0.05, "metadata": "..."}, ...]

// Save / Load
const snapshot = db.save();
const restored = VectorDB.load(snapshot);
```

## HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Max bidirectional connections per node |
| `efConstruction` | 128 | Search width during index building |
| `efSearch` | 40 | Search width during queries (tune for recall/speed) |

Values follow industry consensus from [arxiv:2405.17813](https://arxiv.org/abs/2405.17813).

## Architecture

```
Browser/Node.js → JavaScript API (wasm-bindgen) → WASM (172 KB)
                                                    ├── Distance Metrics (Cosine, L2, IP)
                                                    ├── HNSW Index (multi-layer graph)
                                                    └── Serializer (JSON save/load)
```

## Build from Source

```bash
# Prerequisites: Rust, wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for browser
wasm-pack build --target web --release

# Build for Node.js
wasm-pack build --target nodejs --release

# Run tests
cargo test
```

## License

MIT
