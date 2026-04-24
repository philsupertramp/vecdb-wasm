# VecDB-WASM Documentation

## 📖 Contents

### [Getting Started](./getting-started.md)
Installation, quick start, 5-minute tutorial, and framework integration guides (Vite, Next.js, vanilla).

### [API Reference](./api-reference.md)
Complete reference for every class, method, option, and type:
- **`AgentMemory`** — the primary SDK class (init, mount, search, formatContext, destroy)
- **`VromCache`** — low-level OPFS cache and registry manager
- **`VectorDB`** — the raw WASM HNSW engine
- **Embed Worker Protocol** — message format between main thread and worker
- **Type Index** — all exported TypeScript types

### [Guides](./guides.md)
In-depth guides on key topics:
- Understanding vROMs
- Context expansion strategies
- Building custom vROMs (browser + Python)
- Cache management
- Model diffing and hot-swapping
- Performance tuning (efSearch, token budgets, preloading)
- Error handling patterns
- Python CLI reference

### [Architecture](./architecture.md)
How VecDB-WASM works internally:
- System overview diagram
- HNSW algorithm (parameters, distance metrics, layer structure)
- WASM engine (Rust source, compilation, serialization format, memory model)
- Embed worker (lifecycle, model diffing, zero-copy transfer)
- OPFS cache layer (storage layout, TTL, streamed downloads)
- vROM format specification (manifest, index, chunks)
- Build pipeline (Rust → WASM → TypeScript → npm)
- Design decisions with rationale
