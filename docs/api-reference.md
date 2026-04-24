# API Reference

## Exports

```typescript
// Classes
import { AgentMemory, VromCache } from 'vrom.js';

// Types (type-only imports)
import type {
  AgentMemoryOptions,
  MountOptions,
  MountStatus,
  SearchOptions,
  SearchResult,
  FormatContextOptions,
  ChunkMetadata,
  DownloadProgress,
  StorageEstimate,
  VromRegistry,
  VromRegistryEntry,
  VromManifest,
} from 'vrom.js';
```

---

## Class: `AgentMemory`

The primary SDK class. Wraps the WASM HNSW engine, a background ONNX embedding worker, and an OPFS-backed vROM cache into a single, ergonomic interface.

### Constructor

```typescript
new AgentMemory(options?: AgentMemoryOptions)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `workerPath` | `string` | Auto-resolved via `import.meta.url` | URL to the embed worker JS file. Override when the auto-resolved path is incorrect (e.g., CDN usage). |
| `wasmPkgPath` | `string` | Auto-resolved via `import.meta.url` | URL to the WASM JS bindings module (`vecdb_wasm.js`). |
| `registryUrl` | `string` | HF Hub CDN | URL to a custom vROM registry JSON. Default points to `philipp-zettl/vrom-registry` on HF Hub. |
| `logLevel` | `'silent' \| 'error' \| 'warn' \| 'info' \| 'debug'` | `'warn'` | Console log verbosity. `'info'` is useful during development; `'debug'` logs every download progress event. |

**Example:**

```typescript
// Defaults — works in most bundler setups
const memory = new AgentMemory();

// Explicit paths for CDN or non-bundler setups
const memory = new AgentMemory({
  workerPath: '/static/embed-worker.js',
  wasmPkgPath: '/static/wasm-pkg/vecdb_wasm.js',
  logLevel: 'info',
});
```

---

### `init()`

```typescript
async init(): Promise<void>
```

Initialize the WASM engine and spawn the background embedding worker. **Must be called once before any other method.**

Calling `init()` multiple times is safe — subsequent calls are no-ops.

**What it does:**
1. Dynamically imports the WASM JS bindings module
2. Initializes the WASM binary (`wasm_bindgen init`)
3. Creates a Web Worker from `workerPath` with `type: 'module'`
4. Sets up the internal message handler for the worker protocol

**Throws:**
- If the WASM module fails to load (invalid path, network error)
- If the Worker fails to spawn (CSP violation, invalid path)

**Example:**

```typescript
const memory = new AgentMemory();
await memory.init();
// Now ready to mount vROMs
```

---

### `mount(vromIdOrUri, options?)`

```typescript
async mount(vromIdOrUri: string, options?: MountOptions): Promise<MountStatus>
```

Mount a vROM knowledge base. This is the main loading method that handles the full pipeline: registry lookup → OPFS cache → CDN download → WASM index load → embedding model diffing.

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `vromIdOrUri` | `string` | A vROM identifier. Can be a bare ID (`'hf-transformers-docs'`) or a `hub://` URI (`'hub://hf-transformers-docs'`). |
| `options` | `MountOptions` | Optional configuration (see below). |

**`MountOptions`:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `onProgress` | `(p: DownloadProgress) => void` | — | Callback fired during vROM download. Receives `phase` (`'manifest'`, `'index'`, `'done'`), `file`, `loaded`, and `total` bytes. |
| `forceDownload` | `boolean` | `false` | Skip the OPFS cache and re-download from CDN. Useful for forcing updates. |

**Returns:** `Promise<MountStatus>` — the current state after mounting.

**Mount lifecycle:**

1. **Registry resolve** — looks up the vROM ID in the registry to get CDN URLs, required model, and metadata.
2. **OPFS cache check** — if the vROM is already cached locally, skips the download.
3. **CDN download** (if cache miss) — streams the HNSW index JSON with progress reporting. The manifest and index are written to OPFS.
4. **WASM load** — reads the index JSON from OPFS and calls `VectorDB.load()` to deserialize the HNSW graph into the WASM engine. Frees any previously loaded graph.
5. **Model diffing** — compares the required embedding model (from the vROM manifest) with the currently loaded model. If different, loads the new model in the background worker. If the same, skips reload entirely (hot-swap).

**Throws:**
- `'Call init() first'` — if `init()` hasn't been called
- `'vROM \'...\' not found in registry'` — if the ID doesn't exist in the registry
- `'Failed to read index for \'...\''` — if OPFS read fails after download
- Network errors during CDN download
- Model load errors from the worker

**Example:**

```typescript
// Simple mount
const status = await memory.mount('hf-transformers-docs');
console.log(`${status.vectors} vectors, model: ${status.model}`);

// With progress tracking
await memory.mount('hf-ml-training', {
  onProgress: ({ phase, loaded, total }) => {
    if (phase === 'index' && total > 0) {
      console.log(`Download: ${(loaded / total * 100).toFixed(0)}%`);
    }
  },
  forceDownload: false,
});

// Hot-swap — if the new vROM uses the same embedding model, the model stays loaded
await memory.mount('hf-transformers-docs');
await memory.mount('hf-ml-training'); // Model already loaded → instant swap
```

---

### `unmount()`

```typescript
unmount(): void
```

Unmount the current vROM. Frees the HNSW graph from WASM memory but **keeps the OPFS cache** (so re-mounting is instant). The embedding model stays loaded in the worker.

After unmounting, `search()` will throw until a new vROM is mounted.

---

### `search(query, options?)`

```typescript
async search(query: string, options?: SearchOptions): Promise<SearchResult[]>
```

Search the mounted vROM with a natural language query.

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `query` | `string` | Natural language search query. Gets embedded by the background worker before searching. |
| `options` | `SearchOptions` | Optional search configuration (see below). |

**`SearchOptions`:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `topK` | `number` | `5` | Number of results to return. |
| `expandContext` | `boolean` | `false` | If `true`, follows `prev_chunk_id`/`next_chunk_id` linked-list pointers to expand each result with surrounding chunks. |
| `contextWindow` | `number` | `1` | Number of chunks to expand in each direction (before and after). Only used when `expandContext` is `true`. |
| `efSearch` | `number` | Index default (typically `40`) | Override the HNSW `efSearch` parameter. Higher values increase recall at the cost of speed. Must be ≥ `topK`. |

**Returns:** `Promise<SearchResult[]>` — array of results sorted by distance (ascending, lower = more similar).

**`SearchResult`:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | The chunk text. If `expandContext` is `true`, this is the concatenation of the expanded chunks separated by `\n\n`. |
| `metadata` | `ChunkMetadata & Record<string, any>` | Full chunk metadata including `source_file`, `section_heading`, `url`, `doc_title`, `prev_chunk_id`, `next_chunk_id`. When expanded, also includes `_expanded: true` and `_contextChunks: number`. |
| `distance` | `number` | Cosine distance from the query vector. Range: `[0, 2]`. Lower = more similar. For normalized vectors, `0` = identical, `1` = orthogonal, `2` = opposite. |
| `id` | `number` | Vector ID in the HNSW index. |

**`ChunkMetadata`:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `number` | Unique chunk identifier within the vROM. |
| `text` | `string` | Original chunk text (before context expansion). |
| `source_file` | `string` | Source document filename (e.g., `'transformers/quicktour.md'`). |
| `section_heading` | `string` | Markdown heading for this chunk's section. |
| `prev_chunk_id` | `number \| null` | Previous chunk in the document, or `null` if this is the first chunk. |
| `next_chunk_id` | `number \| null` | Next chunk in the document, or `null` if this is the last chunk. |
| `url` | `string` | Source URL for citation. |
| `doc_title` | `string` | Document title. |
| `_expanded?` | `boolean` | Set to `true` when context expansion was applied. |
| `_contextChunks?` | `number` | Total number of chunks in the expanded text. |

**Throws:**
- `'No vROM mounted — call mount() first'`
- `'Embedding model not loaded'`
- Worker embedding errors

**Example:**

```typescript
// Basic search
const results = await memory.search('how to tokenize text', { topK: 3 });

for (const r of results) {
  console.log(`[${r.distance.toFixed(4)}] ${r.metadata.section_heading}`);
  console.log(r.text.slice(0, 200));
  console.log(`Source: ${r.metadata.url}\n`);
}

// With context expansion — gets surrounding chunks for more context
const expanded = await memory.search('pipeline API', {
  topK: 3,
  expandContext: true,
  contextWindow: 2,  // 2 chunks before + 2 chunks after
});
// Each result.text now contains up to 5 chunks of text
```

---

### `formatContext(results, options?)`

```typescript
formatContext(results: SearchResult[], options?: FormatContextOptions): string
```

Format search results into a string suitable for LLM context/system prompt injection.

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `results` | `SearchResult[]` | Array of search results from `search()`. |
| `options` | `FormatContextOptions` | Optional formatting configuration. |

**`FormatContextOptions`:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `includeSources` | `boolean` | `true` | Append `[Source: <url>]` after each result. |
| `maxTokens` | `number` | `Infinity` | Approximate token budget. Stops adding results when the budget is exceeded. Token count is estimated as `ceil(text.length / 4)`. |

**Returns:** `string` — formatted context, with results separated by `---`.

**Example:**

```typescript
const results = await memory.search('LoRA fine-tuning', { topK: 5 });
const context = memory.formatContext(results, {
  maxTokens: 2000,
  includeSources: true,
});

// Output looks like:
// ## LoRA: Low-Rank Adaptation
// LoRA is a parameter-efficient fine-tuning method...
// [Source: https://huggingface.co/docs/peft/conceptual_guides/lora]
//
// ---
//
// ## Using LoRA with PEFT
// To fine-tune a model with LoRA...
// [Source: https://huggingface.co/docs/peft/tutorial/peft_model_config]
//
// ---
```

---

### `getMountStatus()`

```typescript
getMountStatus(): MountStatus
```

Get the current mount state. Returns a snapshot — the object is not live.

**`MountStatus`:**

| Field | Type | Description |
|-------|------|-------------|
| `activeVrom` | `string \| null` | ID of the currently mounted vROM, or `null`. |
| `version` | `string \| null` | Version string from the vROM manifest. |
| `ready` | `boolean` | `true` if both the HNSW index is loaded and the embedding model is ready. |
| `vectors` | `number` | Number of vectors in the mounted index. `0` if nothing mounted. |
| `dim` | `number` | Vector dimensionality. `0` if nothing mounted. |
| `model` | `string \| null` | Currently loaded embedding model ID (e.g., `'Xenova/all-MiniLM-L6-v2'`). |

---

### `isReady` (getter)

```typescript
get isReady(): boolean
```

Returns `true` if the SDK is fully initialized, a vROM is mounted, and the embedding model is loaded. Equivalent to checking `getMountStatus().ready` after `init()`.

---

### `listVroms()`

```typescript
async listVroms(): Promise<VromRegistryEntry[]>
```

List all available vROMs from the registry. Fetches the registry from CDN on first call, then caches in OPFS for 1 hour.

**`VromRegistryEntry`:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique vROM identifier (e.g., `'hf-transformers-docs'`). |
| `name` | `string` | Human-readable name. |
| `description` | `string` | Short description. |
| `version` | `string` | Semantic version. |
| `vectors` | `number` | Number of vectors in the index. |
| `dimensions` | `number` | Vector dimensionality. |
| `tokens` | `number` | Approximate total tokens in the corpus. |
| `size_mb` | `number` | Download size in megabytes. |
| `model` | `string` | Required embedding model ID. |
| `tags` | `string[]` | Tags for categorization. |
| `official` | `boolean` | Whether this is an official vROM. |
| `files` | `{ manifest: string, index: string, chunks: string }` | CDN URLs for the vROM files. |

**Example:**

```typescript
const vroms = await memory.listVroms();
for (const v of vroms) {
  const cached = await memory.isCached(v.id);
  console.log(`${cached ? '✓' : ' '} ${v.id} — ${v.vectors} vectors, ${v.size_mb} MB`);
}
```

---

### `isCached(vromId)`

```typescript
async isCached(vromId: string): Promise<boolean>
```

Check whether a vROM is cached in OPFS. Returns `true` if the index file exists locally.

---

### `evict(vromId)`

```typescript
async evict(vromId: string): Promise<void>
```

Remove a vROM from the OPFS cache. Deletes all files (manifest, index) for that vROM. Does not affect the currently mounted vROM — call `unmount()` first if evicting the active one.

---

### `storageEstimate()`

```typescript
async storageEstimate(): Promise<StorageEstimate>
```

Get the browser's storage usage estimate.

**`StorageEstimate`:**

| Field | Type | Description |
|-------|------|-------------|
| `used` | `number` | Bytes currently used by the origin. |
| `quota` | `number` | Total bytes available to the origin. |

---

### `onProgress(fn)`

```typescript
onProgress(fn: ((p: { file: string; loaded: number; total: number }) => void) | null): void
```

Set a global progress callback for **embedding model** downloads. This is separate from the `mount()` `onProgress` callback (which tracks vROM index downloads).

The callback fires when the background worker downloads ONNX model files. Pass `null` to remove the callback.

**Example:**

```typescript
memory.onProgress(({ file, loaded, total }) => {
  const pct = total > 0 ? (loaded / total * 100).toFixed(0) : '?';
  updateProgressBar(`${file}: ${pct}%`);
});
```

---

### `destroy()`

```typescript
destroy(): void
```

Destroy the SDK instance. Frees the WASM HNSW graph and terminates the background worker. The OPFS cache is **not** cleared.

After calling `destroy()`, the instance cannot be reused. Create a new `AgentMemory` to start over.

---

## Class: `VromCache`

Low-level OPFS cache and registry manager. Used internally by `AgentMemory`, but exported for advanced use cases.

```typescript
import { VromCache } from 'vrom.js';
```

### Constructor

```typescript
new VromCache(registryUrl?: string)
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `registryUrl` | `string` | HF Hub CDN | Custom registry JSON URL. |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `getRegistry()` | `Promise<VromRegistry>` | Fetch the vROM registry. Caches in OPFS with 1-hour TTL. |
| `resolve(id)` | `Promise<VromRegistryEntry \| null>` | Look up a vROM by ID or `hub://` URI. |
| `list()` | `Promise<VromRegistryEntry[]>` | List all vROMs from the registry. |
| `isCached(id)` | `Promise<boolean>` | Check if a vROM index is in OPFS. |
| `getCachedManifest(id)` | `Promise<VromManifest \| null>` | Read the cached manifest for a vROM. |
| `loadIndex(id)` | `Promise<string \| null>` | Read the raw index JSON from OPFS. |
| `pull(id, entry, onProgress?)` | `Promise<void>` | Download a vROM from CDN and write to OPFS. |
| `evict(id)` | `Promise<void>` | Delete a vROM from OPFS. |
| `storageEstimate()` | `Promise<StorageEstimate>` | Get browser storage usage. |

**`VromManifest`:**

| Field | Type | Description |
|-------|------|-------------|
| `vrom_id` | `string` | Unique identifier. |
| `version` | `string` | Semantic version. |
| `description` | `string` | Human-readable description. |
| `source` | `string` | Source description. |
| `embedding_spec` | `object` | Embedding configuration (see below). |
| `hnsw_config` | `object` | HNSW parameters used to build the index. |
| `vector_count` | `number` | Number of vectors. |
| `total_tokens` | `number` | Approximate token count across all chunks. |
| `total_chunks` | `number` | Number of chunks. |
| `corpus_hash` | `string` | SHA-256 hash prefix of the corpus text. |
| `created_at` | `string` | ISO 8601 build timestamp. |
| `chunk_strategy` | `object` | Chunking parameters used during build. |

**`embedding_spec`:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `string` | HF model ID for the browser (e.g., `'Xenova/all-MiniLM-L6-v2'`). |
| `model_source?` | `string` | Original model source (e.g., `'sentence-transformers/all-MiniLM-L6-v2'`). |
| `dimensions` | `number` | Embedding dimensionality. |
| `quantization` | `string` | Quantization format (e.g., `'q8'`). |
| `distance_metric` | `string` | Distance metric (e.g., `'cosine'`). |
| `normalized` | `boolean` | Whether embeddings are L2-normalized. |

---

## Low-Level: `VectorDB` (WASM)

The raw WASM HNSW engine, exposed via `wasm-bindgen`. This is what `AgentMemory` uses internally. You can use it directly for custom vector search without the vROM/worker layer.

> **Note:** This is exposed from the `wasm-pkg/` directory, not from the main `vrom.js` package export. It's documented here for completeness.

### Constructor

```typescript
new VectorDB(
  dim: number,
  metric?: string | null,     // 'cosine' | 'euclidean' | 'dot_product' (default: 'cosine')
  m?: number | null,           // HNSW M parameter (default: 16)
  ef_construction?: number | null,  // (default: 128)
  ef_search?: number | null,       // (default: 40)
): VectorDB
```

### Static Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `VectorDB.load(json)` | `load(json: string): VectorDB` | Deserialize a VectorDB from JSON. This is how vROM indexes are loaded. |

### Instance Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `insert(vector, metadata?)` | `insert(v: Float32Array, m?: string \| null): number` | Insert a vector with optional JSON metadata. Returns the assigned ID. |
| `insert_batch(vectors, n)` | `insert_batch(v: Float32Array, n: number): number` | Insert `n` vectors from a flat Float32Array. Returns the first assigned ID. |
| `search(query, k)` | `search(q: Float32Array, k: number): string` | Search for `k` nearest neighbors. Returns JSON string of `[{id, distance, metadata}]`. |
| `search_with_ef(query, k, ef)` | `search_with_ef(q: Float32Array, k: number, ef: number): string` | Search with a custom `efSearch` parameter. |
| `get_vector(id)` | `get_vector(id: number): Float32Array \| undefined` | Retrieve the vector for a given ID. |
| `get_metadata(id)` | `get_metadata(id: number): string \| undefined` | Retrieve the metadata JSON string for a given ID. |
| `len()` | `len(): number` | Number of vectors in the index. |
| `dim()` | `dim(): number` | Vector dimensionality. |
| `stats()` | `stats(): string` | JSON string with index statistics. |
| `save()` | `save(): string` | Serialize the entire index to JSON. |
| `free()` | `free(): void` | Free WASM memory. Call when done with the index. |

**`stats()` output:**

```json
{
  "num_vectors": 1356,
  "dimensions": 384,
  "max_layer": 3,
  "total_connections": 42680,
  "avg_connections_per_node": 31.47,
  "memory_bytes": 2456320
}
```

---

## Embed Worker Protocol

The background worker communicates with the main thread via `postMessage`. All outgoing messages are tagged with `source: 'vecdb'` to avoid collisions with transformers.js internal messages.

### Main → Worker

| Message | Fields | Description |
|---------|--------|-------------|
| `load` | `{ type: 'load', modelId: string, dtype: string }` | Load an embedding model. If the same model is already loaded, returns immediately with `cached: true`. |
| `embed` | `{ type: 'embed', texts: string[], id: string }` | Embed a batch of texts. The `id` is used to correlate responses. |
| `unload` | `{ type: 'unload' }` | Dispose the current model and free memory. |
| `get-model` | `{ type: 'get-model' }` | Query the currently loaded model. |

### Worker → Main

| Message | Fields | Description |
|---------|--------|-------------|
| `ready` | `{ status: 'ready', dim, modelId, dtype, cached, source: 'vecdb' }` | Model is loaded and ready. `dim` is the embedding dimensionality. |
| `result` | `{ status: 'result', id, embeddings: Float32Array, dims, source: 'vecdb' }` | Embedding result. `embeddings` is transferred (zero-copy). |
| `dl-progress` | `{ status: 'dl-progress', file, loaded, total, source: 'vecdb' }` | Model download progress. |
| `error` | `{ status: 'error', id?, message, source: 'vecdb' }` | Error during load or embed. |
| `unloaded` | `{ status: 'unloaded', source: 'vecdb' }` | Model successfully unloaded. |
| `model-info` | `{ status: 'model-info', modelId, dtype, dim, loaded, source: 'vecdb' }` | Response to `get-model`. |

---

## Type Index

All types exported from the package:

| Type | Category | Description |
|------|----------|-------------|
| `AgentMemoryOptions` | Config | Constructor options for `AgentMemory` |
| `MountOptions` | Config | Options for `mount()` |
| `SearchOptions` | Config | Options for `search()` |
| `FormatContextOptions` | Config | Options for `formatContext()` |
| `MountStatus` | State | Current mount state snapshot |
| `SearchResult` | Result | Single search result with text, metadata, and distance |
| `ChunkMetadata` | Result | Metadata fields stored per chunk in the HNSW index |
| `DownloadProgress` | Event | Progress event during vROM download |
| `StorageEstimate` | State | Browser storage usage |
| `VromRegistry` | Registry | Full registry object |
| `VromRegistryEntry` | Registry | Single vROM entry from the registry |
| `VromManifest` | Registry | Full manifest stored inside a vROM |
