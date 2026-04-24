/**
 * vecdb-sdk.js — VecDB-WASM Integration SDK
 *
 * Provides the AgentMemory class: a unified, developer-friendly interface
 * for browser-based RAG with zero-latency HNSW search, background embedding,
 * OPFS-cached vROM cartridges, and context hot-swapping.
 *
 * @example
 * ```js
 * import { AgentMemory } from './sdk/vecdb-sdk.js';
 *
 * const memory = new AgentMemory();
 * await memory.init();
 * await memory.mount('hf-transformers-docs');
 * const results = await memory.search('how to use pipelines', { topK: 3 });
 * ```
 */

import { VromCache } from './vrom-cache.js';

// ─── Log Levels ───────────────────────────────────────────────────────────────
const LOG_LEVELS = { silent: 0, error: 1, warn: 2, info: 3, debug: 4 };

/**
 * @typedef {Object} MountStatus
 * @property {string|null} activeVrom - Currently mounted vROM ID
 * @property {string|null} version - vROM version
 * @property {boolean} ready - Whether the vROM is searchable
 * @property {number} vectors - Number of vectors in the index
 * @property {number} dim - Vector dimensionality
 * @property {string|null} model - Embedding model in use
 */

/**
 * @typedef {Object} SearchResult
 * @property {string} text - The chunk text
 * @property {Object} metadata - Full chunk metadata (source_file, section_heading, url, etc.)
 * @property {number} distance - Cosine distance (lower = more similar)
 * @property {number} id - Vector ID in the index
 */

/**
 * @typedef {Object} SearchOptions
 * @property {number} [topK=5] - Number of results
 * @property {boolean} [expandContext=false] - Follow prev/next chunk pointers
 * @property {number} [contextWindow=1] - Chunks to expand in each direction
 * @property {number} [efSearch] - Override HNSW efSearch parameter
 */

export class AgentMemory {
    // ─── Private State ─────────────────────────────────────────────────

    /** @type {object|null} */ #db = null;           // VectorDB WASM instance
    /** @type {Worker|null} */ #worker = null;        // Embed worker
    /** @type {Map} */         #pending = new Map();  // Promise callbacks
    /** @type {VromCache} */   #cache;                // OPFS vROM cache
    /** @type {Function} */    #VectorDB = null;      // VectorDB constructor (from WASM)
    /** @type {Function} */    #initWasm = null;       // WASM init function

    // State tracking
    /** @type {boolean} */ #initialized = false;
    /** @type {boolean} */ #modelReady = false;
    /** @type {number|null} */ #embeddingDim = null;
    /** @type {string|null} */ #currentModelId = null;
    /** @type {string|null} */ #currentDtype = null;
    /** @type {string|null} */ #activeVromId = null;
    /** @type {object|null} */ #activeManifest = null;
    /** @type {number} */      #logLevel;

    // Config
    /** @type {string} */ #workerPath;
    /** @type {string} */ #wasmPkgPath;
    /** @type {string} */ #registryUrl;

    /**
     * Create an AgentMemory instance.
     * @param {Object} [options]
     * @param {string} [options.workerPath='./sdk/embed-worker.js'] - Path to embed worker
     * @param {string} [options.wasmPkgPath='./pkg/vecdb_wasm.js'] - Path to WASM JS bindings
     * @param {string} [options.registryUrl] - Custom vROM registry URL
     * @param {string} [options.logLevel='warn'] - Log level: silent|error|warn|info|debug
     */
    constructor(options = {}) {
        this.#workerPath = options.workerPath || '../sdk/embed-worker.js';
        this.#wasmPkgPath = options.wasmPkgPath || '../pkg/vecdb_wasm.js';
        this.#registryUrl = options.registryUrl;
        this.#logLevel = LOG_LEVELS[options.logLevel || 'warn'] ?? LOG_LEVELS.warn;
        this.#cache = new VromCache(this.#registryUrl);
    }

    // ─── Logging ───────────────────────────────────────────────────────

    #log(level, ...args) {
        if (LOG_LEVELS[level] <= this.#logLevel) {
            const prefix = `[AgentMemory:${level}]`;
            if (level === 'error') console.error(prefix, ...args);
            else if (level === 'warn') console.warn(prefix, ...args);
            else console.log(prefix, ...args);
        }
    }

    // ─── Worker Communication ──────────────────────────────────────────

    #setupWorker() {
        this.#worker = new Worker(this.#workerPath, { type: 'module' });

        this.#worker.addEventListener('message', (e) => {
            const d = e.data;
            if (d.source !== 'vecdb') return;

            switch (d.status) {
                case 'dl-progress':
                    this.#log('debug', `Download: ${d.file} ${((d.loaded / d.total) * 100).toFixed(0)}%`);
                    // Emit event for UI binding
                    this._onProgress?.(d);
                    break;

                case 'ready':
                    this.#embeddingDim = d.dim;
                    this.#currentModelId = d.modelId;
                    this.#currentDtype = d.dtype;
                    this.#modelReady = true;
                    this.#log('info', `Model ready: ${d.modelId} (${d.dim}d)${d.cached ? ' [cached]' : ''}`);
                    this.#resolvePending('__load__');
                    break;

                case 'result':
                    if (this.#pending.has(d.id)) {
                        this.#pending.get(d.id).resolve({ data: d.embeddings, dims: d.dims });
                        this.#pending.delete(d.id);
                    }
                    break;

                case 'unloaded':
                    this.#modelReady = false;
                    this.#embeddingDim = null;
                    this.#currentModelId = null;
                    this.#currentDtype = null;
                    this.#log('info', 'Model unloaded');
                    this.#resolvePending('__unload__');
                    break;

                case 'model-info':
                    this.#resolvePending('__model-info__', d);
                    break;

                case 'error':
                    this.#log('error', d.message);
                    if (d.id && this.#pending.has(d.id)) {
                        this.#pending.get(d.id).reject(new Error(d.message));
                        this.#pending.delete(d.id);
                    }
                    this.#rejectPending('__load__', d.message);
                    break;
            }
        });
    }

    #resolvePending(key, value) {
        if (this.#pending.has(key)) {
            this.#pending.get(key).resolve(value);
            this.#pending.delete(key);
        }
    }

    #rejectPending(key, message) {
        if (this.#pending.has(key)) {
            this.#pending.get(key).reject(new Error(message));
            this.#pending.delete(key);
        }
    }

    #workerCall(key, msg) {
        return new Promise((resolve, reject) => {
            this.#pending.set(key, { resolve, reject });
            this.#worker.postMessage(msg);
        });
    }

    // ─── Embedding (via worker) ────────────────────────────────────────

    /**
     * Embed texts in the background worker. Returns flat Float32Array + dims.
     * @param {string[]} texts
     * @returns {Promise<{data: Float32Array, dims: number[]}>}
     */
    async #embed(texts) {
        if (!this.#modelReady) throw new Error('No embedding model loaded');
        const id = crypto.randomUUID();
        return new Promise((resolve, reject) => {
            this.#pending.set(id, { resolve, reject });
            this.#worker.postMessage({ type: 'embed', texts, id });
        });
    }

    // ─── Public API ────────────────────────────────────────────────────

    /**
     * Initialize the SDK: load WASM binary and spawn the embed worker.
     * Must be called once before any other method.
     */
    async init() {
        if (this.#initialized) return;

        this.#log('info', 'Initializing...');

        // Dynamic import of WASM bindings
        const wasm = await import(this.#wasmPkgPath);
        await wasm.default();
        this.#VectorDB = wasm.VectorDB;
        this.#initWasm = wasm.default;

        // Spawn worker
        this.#setupWorker();

        this.#initialized = true;
        this.#log('info', 'Initialized (WASM + Worker ready)');
    }

    /**
     * Mount a vROM cartridge by ID or hub:// URI.
     * Handles: OPFS cache check → CDN fetch if miss → load into WASM → model diffing.
     *
     * @param {string} vromIdOrUri - e.g. 'hf-transformers-docs' or 'hub://hf-ml-training'
     * @param {Object} [options]
     * @param {function} [options.onProgress] - Progress callback for download
     * @param {boolean} [options.forceDownload=false] - Skip cache, re-fetch from CDN
     * @returns {Promise<MountStatus>}
     */
    async mount(vromIdOrUri, options = {}) {
        if (!this.#initialized) throw new Error('Call init() first');

        const vromId = vromIdOrUri.replace(/^hub:\/\//, '');
        this.#log('info', `Mounting: ${vromId}`);

        // 1. Resolve from registry
        const entry = await this.#cache.resolve(vromId);
        if (!entry) throw new Error(`vROM '${vromId}' not found in registry`);

        // 2. Check OPFS cache
        const cached = !options.forceDownload && await this.#cache.isCached(vromId);

        if (!cached) {
            // 3a. Cache miss → download from CDN
            this.#log('info', `Cache miss — downloading ${vromId} (${entry.size_mb} MB)...`);
            await this.#cache.pull(vromId, entry, options.onProgress);
            this.#log('info', `Downloaded and cached: ${vromId}`);
        } else {
            this.#log('info', `Cache hit: ${vromId}`);
        }

        // 4. Load index from OPFS into WASM
        const indexJson = await this.#cache.loadIndex(vromId);
        if (!indexJson) throw new Error(`Failed to read index for '${vromId}' from OPFS`);

        // Flush existing graph
        if (this.#db) {
            try { this.#db.free(); } catch {}
            this.#db = null;
        }

        this.#db = this.#VectorDB.load(indexJson);
        this.#activeVromId = vromId;
        this.#activeManifest = await this.#cache.getCachedManifest(vromId) || {};

        this.#log('info', `Loaded: ${this.#db.len()} vectors, ${this.#db.dim()}d`);

        // 5. Model diffing — ensure the right embedding model is loaded
        const requiredModel = this.#activeManifest.embedding_spec?.model || entry.model;
        const requiredDtype = this.#activeManifest.embedding_spec?.quantization || 'q8';

        if (!this.#modelReady || this.#currentModelId !== requiredModel || this.#currentDtype !== requiredDtype) {
            this.#log('info', `Model diff: need ${requiredModel} (${requiredDtype}), loading...`);
            await this.#workerCall('__load__', { type: 'load', modelId: requiredModel, dtype: requiredDtype });
        } else {
            this.#log('info', `Model already loaded: ${requiredModel} — skipping reload`);
        }

        return this.getMountStatus();
    }

    /**
     * Unmount the current vROM. Frees WASM memory but keeps OPFS cache.
     */
    unmount() {
        if (this.#db) {
            try { this.#db.free(); } catch {}
            this.#db = null;
        }
        const prev = this.#activeVromId;
        this.#activeVromId = null;
        this.#activeManifest = null;
        this.#log('info', `Unmounted: ${prev}`);
    }

    /**
     * Search the mounted vROM.
     * @param {string} query - Natural language query
     * @param {SearchOptions} [options]
     * @returns {Promise<SearchResult[]>}
     */
    async search(query, options = {}) {
        if (!this.#db) throw new Error('No vROM mounted — call mount() first');
        if (!this.#modelReady) throw new Error('Embedding model not loaded');

        const topK = options.topK || 5;
        const expandContext = options.expandContext || false;
        const contextWindow = options.contextWindow || 1;

        // Embed query in worker
        const output = await this.#embed([query]);
        const queryVec = new Float32Array(output.data.slice(0, this.#embeddingDim));

        // HNSW search
        const rawJson = options.efSearch
            ? this.#db.search_with_ef(queryVec, topK, options.efSearch)
            : this.#db.search(queryVec, topK);

        const rawResults = JSON.parse(rawJson);

        // Parse and expand
        const results = rawResults.map(r => {
            const meta = r.metadata ? JSON.parse(r.metadata) : {};
            return {
                text: meta.text || '',
                metadata: meta,
                distance: r.distance,
                id: r.id,
            };
        });

        // Context expansion via linked-list pointers
        if (expandContext) {
            for (const result of results) {
                const before = [];
                const after = [];

                // Walk backwards
                let prevId = result.metadata.prev_chunk_id;
                for (let i = 0; i < contextWindow && prevId != null; i++) {
                    const raw = this.#db.get_metadata(prevId);
                    if (!raw) break;
                    const m = JSON.parse(raw);
                    before.unshift(m.text || '');
                    prevId = m.prev_chunk_id;
                }

                // Walk forwards
                let nextId = result.metadata.next_chunk_id;
                for (let i = 0; i < contextWindow && nextId != null; i++) {
                    const raw = this.#db.get_metadata(nextId);
                    if (!raw) break;
                    const m = JSON.parse(raw);
                    after.push(m.text || '');
                    nextId = m.next_chunk_id;
                }

                if (before.length || after.length) {
                    result.text = [...before, result.text, ...after].join('\n\n');
                    result.metadata._expanded = true;
                    result.metadata._contextChunks = before.length + 1 + after.length;
                }
            }
        }

        return results;
    }

    /**
     * Format search results as a context string for LLM injection.
     * @param {SearchResult[]} results
     * @param {Object} [options]
     * @param {boolean} [options.includeSources=true] - Include source URLs
     * @param {number} [options.maxTokens] - Approximate token budget
     * @returns {string}
     */
    formatContext(results, options = {}) {
        const includeSources = options.includeSources !== false;
        const maxTokens = options.maxTokens || Infinity;

        let context = '';
        let approxTokens = 0;

        for (const r of results) {
            const chunk = r.text;
            const chunkTokens = Math.ceil(chunk.length / 4);

            if (approxTokens + chunkTokens > maxTokens) break;

            context += chunk + '\n';
            if (includeSources && r.metadata.url) {
                context += `[Source: ${r.metadata.url}]\n`;
            }
            context += '\n---\n\n';
            approxTokens += chunkTokens;
        }

        return context.trim();
    }

    // ─── State Queries ─────────────────────────────────────────────────

    /**
     * Get the current mount status.
     * @returns {MountStatus}
     */
    getMountStatus() {
        return {
            activeVrom: this.#activeVromId,
            version: this.#activeManifest?.version || null,
            ready: !!this.#db && this.#modelReady,
            vectors: this.#db ? this.#db.len() : 0,
            dim: this.#db ? this.#db.dim() : 0,
            model: this.#currentModelId,
        };
    }

    /**
     * Check if the SDK is fully initialized and ready.
     * @returns {boolean}
     */
    get isReady() {
        return this.#initialized && !!this.#db && this.#modelReady;
    }

    /**
     * List all available vROMs from the registry.
     * @returns {Promise<object[]>}
     */
    async listVroms() {
        return this.#cache.list();
    }

    /**
     * Check if a vROM is cached locally.
     * @param {string} vromId
     * @returns {Promise<boolean>}
     */
    async isCached(vromId) {
        return this.#cache.isCached(vromId);
    }

    /**
     * Evict a vROM from the local OPFS cache.
     * @param {string} vromId
     */
    async evict(vromId) {
        await this.#cache.evict(vromId);
        this.#log('info', `Evicted from cache: ${vromId}`);
    }

    /**
     * Get local storage usage.
     * @returns {Promise<{used: number, quota: number}>}
     */
    async storageEstimate() {
        return this.#cache.storageEstimate();
    }

    /**
     * Set a progress callback for model downloads (for UI binding).
     * @param {function|null} fn - ({file, loaded, total}) => void
     */
    onProgress(fn) {
        this._onProgress = fn;
    }

    /**
     * Destroy the SDK instance. Frees WASM memory and terminates the worker.
     */
    destroy() {
        if (this.#db) { try { this.#db.free(); } catch {} }
        if (this.#worker) this.#worker.terminate();
        this.#db = null;
        this.#worker = null;
        this.#initialized = false;
        this.#modelReady = false;
        this.#log('info', 'Destroyed');
    }
}
