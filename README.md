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

# 🔍 VecDB-WASM

A WebAssembly vector search database with a zero-boilerplate SDK for browser-based RAG. Sub-millisecond HNSW search, background ONNX embedding, OPFS-cached vROM cartridges, and context hot-swapping — 100% client-side.

## 📖 Documentation

| | |
|---|---|
| **[Getting Started](./docs/getting-started.md)** | Installation, quick start, 5-minute tutorial, framework integration |
| **[API Reference](./docs/api-reference.md)** | Every class, method, option, and type — exhaustively documented |
| **[Guides](./docs/guides.md)** | vROMs, context expansion, custom knowledge bases, Python CLI, performance tuning |
| **[Architecture](./docs/architecture.md)** | HNSW internals, WASM engine, worker protocol, OPFS cache, vROM format spec |

## Repository Layout

This is a **source-only repository**. All build artifacts (`pkg/`, `lib/wasm-pkg/`, `lib/dist/`) are generated from the source files below.

```
src/                           ← Rust WASM engine (source of truth)
  Cargo.toml                      Crate manifest
  rust/
    lib.rs                        wasm-bindgen API (VectorDB class)
    hnsw.rs                       HNSW algorithm implementation
    distance.rs                   Cosine / Euclidean / Dot Product metrics

lib/                           ← npm package (TypeScript SDK)
  src/
    index.ts                      Barrel export
    agent-memory.ts               AgentMemory class (init/mount/search)
    vrom-cache.ts                 OPFS cache + hub:// URI resolver
    embed-worker.ts               Background ONNX worker (model diffing)
    types.ts                      All TypeScript type definitions
  package.json                    npm package config + build scripts
  tsconfig.json                   TypeScript compiler config
  tsdown.config.ts                Build config (ESM + CJS + .d.ts + worker)
  LICENSE

docs/                          ← Documentation
  getting-started.md              Install, quick start, framework guides
  api-reference.md                Full API reference
  guides.md                       In-depth guides
  architecture.md                 System internals

tools/                         ← Python tooling
  vrom_builder.py                 Build vROM packages from documentation
  vrom_cli.py                     CLI: list / pull / search / info

index.html                     ← Space UI (Engine + vROM Hub + Builder)
embed-worker.js                   Worker for the Space UI
pkg/                              Pre-built WASM bindings for the Space UI
```

## Quick Start

```bash
npm install vecdb-wasm
```

```typescript
import { AgentMemory } from 'vecdb-wasm';

const memory = new AgentMemory();
await memory.init();

// Mount a vROM — auto-caches to OPFS, auto-loads embedding model
await memory.mount('hf-transformers-docs');

// Search with context expansion
const results = await memory.search("how to use pipelines", {
    topK: 3,
    expandContext: true,
});

// Format for LLM injection
const context = memory.formatContext(results, { maxTokens: 2000 });

// Hot-swap to a different domain — skips model reload if compatible
await memory.mount('hf-ml-training');
```

→ See **[Getting Started](./docs/getting-started.md)** for the full tutorial, and **[API Reference](./docs/api-reference.md)** for all methods and options.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Rust](https://rustup.rs/) | stable | Compile HNSW engine to WASM |
| [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) | ≥0.13 | Rust → WASM + JS bindings |
| [Node.js](https://nodejs.org/) | ≥22 | Build TypeScript SDK |
| [Python](https://python.org/) | ≥3.10 | vROM builder + CLI (optional) |

```bash
# Install Rust + wasm-pack (if not already)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install wasm-pack
```

## Building from Source

The build has two stages: **Rust → WASM**, then **TypeScript → npm package**.

### Quick Build

```bash
cd lib
npm install
npm run build        # runs build:wasm then build:js
```

This produces:
```
lib/wasm-pkg/        ← wasm-pack output (JS bindings + .wasm binary)
lib/dist/            ← Final npm package contents
  index.js             ESM  (14 KB)
  index.cjs            CJS  (15 KB)
  index.d.ts           TypeScript declarations
  index.d.cts          TypeScript declarations (CJS)
  embed-worker.js      Background worker (separate file)
  embed-worker.d.ts    Worker type declarations
  vecdb_wasm_bg.wasm   WASM binary (172 KB)
```

### Stage 1: Compile Rust → WASM

```bash
cd lib
npm run build:wasm
```

Runs `wasm-pack build ../src --target bundler --out-dir ../lib/wasm-pkg --release`.

### Stage 2: Bundle TypeScript → npm package

```bash
cd lib
npm run build:js
```

Runs `tsdown` which bundles ESM + CJS + type declarations + worker + WASM copy.

### Validation

```bash
cd lib
npm run build:check   # TypeScript type-check (no emit)
npm run lint          # publint — validates exports map, file extensions
npm run lint:types    # are-the-types-wrong — checks type resolution
npm pack --dry-run    # Preview what npm publish would include
```

### Build Scripts Reference

| Script | Command | What it does |
|--------|---------|--------------|
| `build:wasm` | `wasm-pack build ...` | Rust → WASM + JS bindings |
| `build:js` | `tsdown` | TypeScript → ESM/CJS/dts + worker + WASM copy |
| `build` | `build:wasm && build:js` | Full pipeline |
| `build:check` | `tsc --noEmit` | Type-check only |
| `lint` | `publint` | Validate package exports |
| `lint:types` | `attw --pack .` | Check type resolution across CJS/ESM |
| `clean` | `rm -rf dist wasm-pkg` | Remove all build artifacts |
| `prepack` | `npm run build` | Auto-runs before `npm pack` / `npm publish` |

## CI/CD

```yaml
name: Build & Publish

on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
      - uses: jetli/wasm-pack-action@v0.4.0
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          registry-url: https://registry.npmjs.org

      - name: Install dependencies
        run: cd lib && npm ci

      - name: Build (Rust → WASM → TypeScript)
        run: cd lib && npm run build

      - name: Validate
        run: cd lib && npm run lint

      - name: Publish to npm
        run: cd lib && npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

## API Overview

| Method | Returns | Description |
|--------|---------|-------------|
| `init()` | `Promise<void>` | Initialize WASM + spawn worker |
| `mount(id, opts?)` | `Promise<MountStatus>` | Mount a vROM (OPFS cache → CDN fallback → model diff) |
| `unmount()` | `void` | Free HNSW graph from RAM |
| `search(query, opts?)` | `Promise<SearchResult[]>` | Embed + HNSW search + context expansion |
| `formatContext(results, opts?)` | `string` | Format as LLM context string |
| `getMountStatus()` | `MountStatus` | Current state |
| `listVroms()` | `Promise<VromRegistryEntry[]>` | List available vROMs |
| `isCached(id)` | `Promise<boolean>` | Check OPFS cache |
| `evict(id)` | `Promise<void>` | Remove from OPFS cache |
| `destroy()` | `void` | Free all resources + terminate worker |

→ See **[API Reference](./docs/api-reference.md)** for full parameter documentation, types, and examples.

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
│  │  📡 vROM Registry CDN — HF Hub dataset resolve/ endpoints  │     │
│  └───────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

→ See **[Architecture](./docs/architecture.md)** for the full deep-dive.

## Official vROMs

Pre-computed HNSW indexes served from the [vROM Registry](https://huggingface.co/datasets/philipp-zettl/vrom-registry):

| ID | Vectors | Size | Content |
|----|---------|------|---------|
| `hf-transformers-docs` | 1,356 | 12.6 MB | HF Transformers + Hub docs |
| `hf-ml-training` | 629 | 5.8 MB | TRL + PEFT + Datasets docs |

→ See **[Guides: Understanding vROMs](./docs/guides.md#understanding-vroms)** and **[Guides: Building Custom vROMs](./docs/guides.md#building-custom-vroms)** for more.

## Performance

| Metric | Value |
|--------|-------|
| HNSW Search | < 1 ms |
| Embedding (worker) | ~50 ms/sentence |
| vROM mount (cached) | < 500 ms |
| Hot-swap (same model) | < 500 ms |
| WASM Binary | 172 KB |
| npm tarball | 178 KB |

→ See **[Guides: Performance Tuning](./docs/guides.md#performance-tuning)** for efSearch tradeoffs and optimization tips.

## License

MIT
