# vROM.js

A high-performance, local-first vector database compiled to WebAssembly. 
`vROM.js` provides seamless LLM context monitoring and agent memory management. 
It allows you to mount and query local AI knowledge cartridges directly in Node.js or the browser.

## 🚀 Features

* **Universal Compatibility:** Native support for both Node.js (`vrom.js`) and Web environments (`vrom.js/web`).
* **WebAssembly Native:** Core HNSW indexing and distance calculations are written in Rust and compiled to Wasm for near-native computational speeds.
* **Non-Blocking Architecture:** Offloads heavy embedding and search computations using dedicated Web Workers in the browser.
* **Agent Memory Management:** First-class TypeScript support for context expansion, chunk tracking, and persistent state saving.

## 📦 Installation

```bash
npm install vrom.js

```

## 🗂️ Knowledge Cartridges & Registries

By default, `vROM.js` resolves and fetches knowledge cartridges from the official central dataset registry:

👉 **Default Registry:** [philipp-zettl/vrom-registry on Hugging Face](https://huggingface.co/datasets/philipp-zettl/vrom-registry)

When you call `.mount('cartridge-name')`, the library looks for pre-built vector cartridges hosted within this registry.

## 🛠️ Usage

### Node.js (Native)

For server-side or local script usage, import the default Node native variant:

```typescript
import { AgentMemory } from 'vrom.js';

const instance = new AgentMemory();
await instance.init();

// Mounts 'hf-inference-docs' from the default Hugging Face registry
await instance.mount('hf-inference-docs');

```

### Web (Browser)

When using `vROM.js` in a web environment, you must ensure the embedding Web Worker is available in your public directory so the browser can serve it.

#### 1. Make the Worker Available

**Option A: Automated Setup (Recommended)**
Automate the copy step by adding `prestart` and `prebuild` scripts to your `package.json` (example using `react-scripts`):

```json
"scripts": {
    "prestart": "cp node_modules/vrom.js/dist/embed-worker.js public/ || true",
    "prebuild": "cp node_modules/vrom.js/dist/embed-worker.js public/ || true",
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
}

```

*(Note: If using Vite or another bundler, adjust `public/` to match your static asset directory).*

**Option B: Manual Copy**

```bash
cp node_modules/vrom.js/dist/embed-worker.js public/

```

#### 2. Initialize the Web Variant

```typescript
import { AgentMemory } from 'vrom.js/web';

const instance = new AgentMemory({ workerPath: '/embed-worker.js' });
await instance.init();

// Mount a knowledge cartridge from the registry
await instance.mount('hf-inference-docs');

```

### Configuration Options
To configure the agent we have the following options available
```typescript
export interface AgentMemoryOptions {
    /** Path to the embed worker JS file. Default: auto-resolved via import.meta.url */
    workerPath?: string;
    /** Path to the WASM JS bindings module. Default: auto-resolved via import.meta.url */
    wasmPkgPath?: string;
    /** Custom vROM registry URL. Default: HF Hub CDN */
    registryUrl?: string;
    /** Log level. Default: 'warn' */
    logLevel?: 'silent' | 'error' | 'warn' | 'info' | 'debug';

    /* Dedicated field for SaaS authentication.
     * Automatically maps to the 'x-api-key' header.
     */
    apiKey?: string;

    /* Custom headers for all registry and vROM asset requests
     * Useful for custom proxy auth, User-Agent, etc.
     */
    headers?: Record<string, string> | Headers;
}
```

When mounting a ROM we can provide the following options
```typescript
export interface MountOptions {
    /** Progress callback for CDN download. */
    onProgress?: (progress: DownloadProgress) => void;
    /** Force re-download even if cached in OPFS. Default: false */
    forceDownload?: boolean;
}
```

## 📖 API Reference

Once your `AgentMemory` instance is initialized, you have access to a full suite of vector operations.

### Insert Data

Add text and rich metadata to the memory instance:

```typescript
await instance.insert("My text string", { id: "VEC_ID", key: "value", meta: "data" });

```

### Delete Data

Remove a specific document by its Vector ID:

```typescript
await instance.deleteDoc("VEC_ID");

```

### Generate Embeddings

Directly embed text queries using the underlying model:

```typescript
const embeddings = await instance.embed(["my query"]);

```

### Search

Perform approximate nearest neighbor searches against the memory:

```typescript
const results = await instance.search("my query", { topK: 5 });

```

**Search Options:**
You can tune the search behavior and enable context expansion using the `SearchOptions` interface:

```typescript
export interface SearchOptions {
    /** Number of results. Default: 5 */
    topK?: number;
    /** Follow prev/next chunk pointers for context expansion. Default: false */
    expandContext?: boolean;
    /** Number of chunks to expand in each direction. Default: 1 */
    contextWindow?: number;
    /** Override HNSW efSearch parameter for quality/speed tradeoff. */
    efSearch?: number;
}

```

### Save State

Export the current agent memory content to be stored or shared:

```typescript
const indexString = await instance.save();

```

## 📁 Repository Structure

* **`/src`**: The Rust backend containing the core HNSW index and distance metrics.
* **`/lib`**: The TypeScript library providing the API for web/node clients.
* **`/tools`**: Python scripts (`vrom_cli.py`, `vrom_builder.py`) for building and managing vector data offline.

## 📄 License

This project is licensed under the terms found in the `LICENSE` file.

