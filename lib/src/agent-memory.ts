import { VromCache } from './vrom-cache.js';
import type {
    AgentMemoryOptions,
    MountOptions,
    MountStatus,
    SearchOptions,
    SearchResult,
    FormatContextOptions,
    VectorDB,
    VectorDBConstructor,
    WorkerOutMessage,
    StorageEstimate,
    VromRegistryEntry,
    DownloadProgress,
} from './types.js';
export type WasmLoader = () => Promise<{ VectorDB: VectorDBConstructor }>;

const LOG_LEVELS = { silent: 0, error: 1, warn: 2, info: 3, debug: 4 } as const satisfies Record<string, number>;
type LogLevel = keyof typeof LOG_LEVELS;

/**
 * AgentMemory — zero-boilerplate RAG for browser AI agents.
 *
 * Wraps the VecDB-WASM HNSW engine, a background ONNX embedding worker,
 * and an OPFS-backed vROM cache into a single class.
 *
 * Lifecycle: `constructor` → {@link init} → {@link mount} → {@link search} → {@link destroy}
 *
 * @example
 * ```ts
 * const memory = new AgentMemory();
 * await memory.init();
 * await memory.mount('hf-transformers-docs');
 * const results = await memory.search('how to use pipelines', { topK: 3, expandContext: true });
 * const context = memory.formatContext(results, { maxTokens: 2000 });
 * ```
 */
export class AgentMemoryCore {
    #db: VectorDB | null = null;
    #worker: Worker | null = null;
    #pending = new Map<string, { resolve: (v: any) => void; reject: (e: Error) => void }>();
    #cache: VromCache;
    #VectorDB: VectorDBConstructor | null = null;

    #initialized = false;
    #modelReady = false;
    #embeddingDim: number | null = null;
    #currentModelId: string | null = null;
    #currentDtype: string | null = null;
    #activeVromId: string | null = null;
    #activeManifest: any = null;
    #logLevel: number;

    #workerPath: string;
    #wasmLoader: WasmLoader; // <-- Store the injected loader

    /** Optional progress callback for model downloads. Set via {@link onProgress}. */
    _onProgress?: (p: { file: string; loaded: number; total: number }) => void;

    /**
     * Create an AgentMemory instance.
     *
     * Does not perform any async work — call {@link init} to start the engine.
     *
     * @param options - Configuration options. All fields are optional with sensible defaults.
     */
    constructor(wasmLoader: WasmLoader, options: AgentMemoryOptions = {}) {
        this.#wasmLoader = wasmLoader;
        
        // Remove the #wasmPkgPath fallback logic entirely!
        this.#workerPath = options.workerPath ?? new URL('./embed-worker.js', import.meta.url).href;
        const level = options.logLevel ?? 'warn';
        this.#logLevel = LOG_LEVELS[level];
        this.#cache = new VromCache(options.registryUrl);
    }

    // ─── Logging ───────────────────────────────────────────────────────

    #log(level: LogLevel, ...args: any[]) {
        if (LOG_LEVELS[level] <= this.#logLevel) {
            const prefix = `[AgentMemory:${level}]`;
            if (level === 'error') console.error(prefix, ...args);
            else if (level === 'warn') console.warn(prefix, ...args);
            else console.log(prefix, ...args);
        }
    }

    // ─── Worker ────────────────────────────────────────────────────────

    #setupWorker() {
        this.#worker = new Worker(this.#workerPath, { type: 'module' });

        this.#worker.addEventListener('message', (e: MessageEvent<WorkerOutMessage>) => {
            const d = e.data;
            if ((d as any).source !== 'vecdb') return;

            switch (d.status) {
                case 'dl-progress':
                    this.#log('debug', `DL: ${d.file} ${((d.loaded / d.total) * 100).toFixed(0)}%`);
                    this._onProgress?.(d);
                    break;

                case 'ready':
                    this.#embeddingDim = d.dim;
                    this.#currentModelId = d.modelId;
                    this.#currentDtype = d.dtype;
                    this.#modelReady = true;
                    this.#log('info', `Model ready: ${d.modelId} (${d.dim}d)${d.cached ? ' [cached]' : ''}`);
                    this.#resolve('__load__');
                    break;

                case 'result':
                    if (this.#pending.has(d.id)) {
                        this.#pending.get(d.id)!.resolve({ data: d.embeddings, dims: d.dims });
                        this.#pending.delete(d.id);
                    }
                    break;

                case 'unloaded':
                    this.#modelReady = false;
                    this.#embeddingDim = null;
                    this.#currentModelId = null;
                    this.#currentDtype = null;
                    this.#log('info', 'Model unloaded');
                    this.#resolve('__unload__');
                    break;

                case 'model-info':
                    this.#resolve('__model-info__', d);
                    break;

                case 'error':
                    this.#log('error', d.message);
                    if (d.id && this.#pending.has(d.id)) {
                        this.#pending.get(d.id)!.reject(new Error(d.message));
                        this.#pending.delete(d.id);
                    }
                    this.#reject('__load__', d.message);
                    break;
            }
        });
    }

    #resolve(key: string, value?: any) {
        if (this.#pending.has(key)) {
            this.#pending.get(key)!.resolve(value);
            this.#pending.delete(key);
        }
    }

    #reject(key: string, message: string) {
        if (this.#pending.has(key)) {
            this.#pending.get(key)!.reject(new Error(message));
            this.#pending.delete(key);
        }
    }

    #workerRPC<T = void>(key: string, msg: any): Promise<T> {
        return new Promise((resolve, reject) => {
            this.#pending.set(key, { resolve, reject });
            this.#worker!.postMessage(msg);
        });
    }

    async #embed(texts: string[]): Promise<{ data: Float32Array; dims: number[] }> {
        if (!this.#modelReady) throw new Error('No embedding model loaded');
        const id = crypto.randomUUID();
        return new Promise((resolve, reject) => {
            this.#pending.set(id, { resolve, reject });
            this.#worker!.postMessage({ type: 'embed', texts, id });
        });
    }

    // ─── Public API ────────────────────────────────────────────────────

    /**
     * Initialize the WASM engine and spawn the background embedding worker.
     *
     * Must be called once before {@link mount}, {@link search}, or any other method.
     * Calling `init()` multiple times is safe — subsequent calls are no-ops.
     *
     * @throws If the WASM module fails to load (invalid path, network error)
     * @throws If the Web Worker fails to spawn (CSP violation, invalid path)
     *
     * @example
     * ```ts
     * const memory = new AgentMemory();
     * await memory.init();
     * ```
     */
    async init(): Promise<void> {
        if (this.#initialized) return;
        this.#log('info', 'Initializing...');

        // Execute the injected loading logic!
        const wasm = await this.#wasmLoader();
        this.#VectorDB = wasm.VectorDB;

        this.#setupWorker();
        this.#initialized = true;
        this.#log('info', 'Initialized');
    }

    /**
     * Mount a vROM cartridge. Handles the full pipeline: registry lookup →
     * OPFS cache check → CDN download → WASM index load → embedding model diffing.
     *
     * If the required embedding model is already loaded from a previous mount,
     * model reload is skipped entirely (hot-swap).
     *
     * @param vromIdOrUri - vROM identifier, e.g. `'hf-transformers-docs'` or `'hub://hf-ml-training'`
     * @param options - Mount options (progress callback, force download)
     * @returns Current state after mounting
     *
     * @throws `'Call init() first'` — if {@link init} hasn't been called
     * @throws `'vROM \'...\' not found in registry'` — if the ID doesn't exist
     * @throws Network errors during CDN download
     *
     * @see {@link unmount} to free the HNSW graph
     * @see {@link getMountStatus} to inspect the current state
     *
     * @example
     * ```ts
     * const status = await memory.mount('hf-transformers-docs', {
     *   onProgress: ({ phase, loaded, total }) => {
     *     if (phase === 'index' && total > 0)
     *       console.log(`${(loaded / total * 100).toFixed(0)}%`);
     *   },
     * });
     * console.log(`${status.vectors} vectors ready`);
     * ```
     */
    async mount(vromIdOrUri: string, options: MountOptions = {}): Promise<MountStatus> {
        if (!this.#initialized) throw new Error('Call init() first');

        const vromId = vromIdOrUri.replace(/^hub:\/\//, '');
        this.#log('info', `Mounting: ${vromId}`);

        const entry = await this.#cache.resolve(vromId);
        if (!entry) throw new Error(`vROM '${vromId}' not found in registry`);

        // OPFS cache check
        const cached = !options.forceDownload && (await this.#cache.isCached(vromId));
        if (!cached) {
            this.#log('info', `Cache miss → downloading ${vromId} (${entry.size_mb} MB)`);
            await this.#cache.pull(vromId, entry, options.onProgress);
        } else {
            this.#log('info', `Cache hit: ${vromId}`);
        }

        // Load into WASM (flush old graph)
        const indexJson = await this.#cache.loadIndex(vromId);
        if (!indexJson) throw new Error(`Failed to read index for '${vromId}'`);

        if (this.#db) { try { this.#db.free(); } catch {} }
        this.#db = this.#VectorDB!.load(indexJson);
        this.#activeVromId = vromId;
        this.#activeManifest = (await this.#cache.getCachedManifest(vromId)) ?? {};

        this.#log('info', `Loaded: ${this.#db.len()} vectors, ${this.#db.dim()}d`);

        // Model diffing
        const requiredModel = this.#activeManifest.embedding_spec?.model || entry.model;
        const requiredDtype = this.#activeManifest.embedding_spec?.quantization || 'q8';

        if (!this.#modelReady || this.#currentModelId !== requiredModel || this.#currentDtype !== requiredDtype) {
            this.#log('info', `Model diff: need ${requiredModel} (${requiredDtype})`);
            await this.#workerRPC('__load__', { type: 'load', modelId: requiredModel, dtype: requiredDtype });
        } else {
            this.#log('info', `Model match: ${requiredModel} — skip reload`);
        }

        return this.getMountStatus();
    }

    /**
     * Unmount the current vROM. Frees the HNSW graph from WASM memory
     * but preserves the OPFS cache (so re-mounting is instant).
     *
     * The embedding model remains loaded in the worker.
     * After unmounting, {@link search} will throw until a new vROM is mounted.
     *
     * @see {@link evict} to also remove from cache
     */
    unmount(): void {
        if (this.#db) { try { this.#db.free(); } catch {} }
        this.#db = null;
        const prev = this.#activeVromId;
        this.#activeVromId = null;
        this.#activeManifest = null;
        this.#log('info', `Unmounted: ${prev}`);
    }

    /**
     * Search the mounted vROM with a natural language query.
     *
     * The query is embedded in the background worker (~50ms), then
     * HNSW approximate nearest neighbor search runs in WASM (<1ms).
     *
     * @param query - Natural language search query
     * @param options - Search configuration (topK, context expansion, efSearch)
     * @returns Results sorted by distance ascending (lower = more similar)
     *
     * @throws `'No vROM mounted — call mount() first'`
     * @throws `'Embedding model not loaded'`
     *
     * @example
     * ```ts
     * const results = await memory.search('how to fine-tune', {
     *   topK: 5,
     *   expandContext: true,
     *   contextWindow: 1,
     * });
     * ```
     */
    async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
        if (!this.#db) throw new Error('No vROM mounted — call mount() first');
        if (!this.#modelReady) throw new Error('Embedding model not loaded');

        const topK = options.topK ?? 5;
        const expandContext = options.expandContext ?? false;
        const contextWindow = options.contextWindow ?? 1;

        const output = await this.#embed([query]);
        const vec = new Float32Array(output.data.slice(0, this.#embeddingDim!));

        const rawJson = options.efSearch
            ? this.#db.search_with_ef(vec, topK, options.efSearch)
            : this.#db.search(vec, topK);

        const results: SearchResult[] = JSON.parse(rawJson).map((r: any) => {
            const meta = r.metadata ? JSON.parse(r.metadata) : {};
            return { text: meta.text ?? '', metadata: meta, distance: r.distance, id: r.id };
        });

        if (expandContext) {
            for (const result of results) {
                const before: string[] = [];
                const after: string[] = [];

                let pid = result.metadata.prev_chunk_id;
                for (let i = 0; i < contextWindow && pid != null; i++) {
                    const raw = this.#db.get_metadata(pid);
                    if (!raw) break;
                    const m = JSON.parse(raw);
                    before.unshift(m.text ?? '');
                    pid = m.prev_chunk_id;
                }

                let nid = result.metadata.next_chunk_id;
                for (let i = 0; i < contextWindow && nid != null; i++) {
                    const raw = this.#db.get_metadata(nid);
                    if (!raw) break;
                    const m = JSON.parse(raw);
                    after.push(m.text ?? '');
                    nid = m.next_chunk_id;
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
     * Format search results as a context string for LLM prompt injection.
     *
     * Concatenates result texts separated by `---` markers. Optionally includes
     * source URLs and respects an approximate token budget.
     *
     * @param results - Search results from {@link search}
     * @param options - Formatting options (sources, token budget)
     * @returns Formatted context string ready for LLM system/user prompt
     *
     * @example
     * ```ts
     * const context = memory.formatContext(results, {
     *   maxTokens: 2000,
     *   includeSources: true,
     * });
     * ```
     */
    formatContext(results: SearchResult[], options: FormatContextOptions = {}): string {
        const includeSources = options.includeSources !== false;
        const maxTokens = options.maxTokens ?? Infinity;

        let ctx = '';
        let tokens = 0;

        for (const r of results) {
            const t = Math.ceil(r.text.length / 4);
            if (tokens + t > maxTokens) break;
            ctx += r.text + '\n';
            if (includeSources && r.metadata.url) ctx += `[Source: ${r.metadata.url}]\n`;
            ctx += '\n---\n\n';
            tokens += t;
        }

        return ctx.trim();
    }

    // ─── Queries ───────────────────────────────────────────────────────

    /**
     * Get a snapshot of the current mount state.
     *
     * @returns A plain object describing the active vROM, model, and readiness.
     * The returned object is not live — it reflects the state at call time.
     */
    getMountStatus(): MountStatus {
        return {
            activeVrom: this.#activeVromId,
            version: this.#activeManifest?.version ?? null,
            ready: !!this.#db && this.#modelReady,
            vectors: this.#db?.len() ?? 0,
            dim: this.#db?.dim() ?? 0,
            model: this.#currentModelId,
        };
    }

    /**
     * Whether the SDK is fully ready: initialized, vROM mounted, and model loaded.
     *
     * @remarks Equivalent to `getMountStatus().ready` after `init()`.
     */
    get isReady(): boolean {
        return this.#initialized && !!this.#db && this.#modelReady;
    }

    /**
     * List all available vROMs from the registry.
     *
     * Fetches the registry from CDN on first call, then caches in OPFS for 1 hour.
     *
     * @returns Array of registry entries with IDs, sizes, model requirements, and CDN URLs
     */
    async listVroms(): Promise<VromRegistryEntry[]> {
        return this.#cache.list();
    }

    /**
     * Check whether a vROM is cached locally in OPFS.
     *
     * @param vromId - vROM identifier
     * @returns `true` if the index file exists in OPFS
     */
    async isCached(vromId: string): Promise<boolean> {
        return this.#cache.isCached(vromId);
    }

    /**
     * Evict a vROM from the OPFS cache.
     *
     * Deletes all cached files (manifest + index) for the given vROM.
     * Does not affect the currently mounted vROM — call {@link unmount} first
     * if evicting the active one.
     *
     * @param vromId - vROM identifier to evict
     */
    async evict(vromId: string): Promise<void> {
        await this.#cache.evict(vromId);
        this.#log('info', `Evicted: ${vromId}`);
    }

    /**
     * Get the browser's storage usage estimate.
     *
     * @returns Used and quota bytes for the current origin
     */
    async storageEstimate(): Promise<StorageEstimate> {
        return this.#cache.storageEstimate();
    }

    /**
     * Set a global progress callback for embedding model downloads.
     *
     * This is separate from the per-mount `onProgress` callback (which tracks
     * vROM index downloads). This callback fires when the background worker
     * downloads ONNX model weight files.
     *
     * @param fn - Progress callback, or `null` to remove
     *
     * @example
     * ```ts
     * memory.onProgress(({ file, loaded, total }) => {
     *   console.log(`${file}: ${(loaded / total * 100).toFixed(0)}%`);
     * });
     * ```
     */
    onProgress(fn: ((p: { file: string; loaded: number; total: number }) => void) | null): void {
        this._onProgress = fn ?? undefined;
    }

    /**
     * Destroy the SDK instance. Frees the WASM HNSW graph and terminates
     * the background embedding worker.
     *
     * The OPFS cache is **not** cleared — cached vROMs persist for future sessions.
     * After calling `destroy()`, the instance cannot be reused.
     *
     * @see {@link evict} to clear specific vROMs from cache
     */
    destroy(): void {
        if (this.#db) { try { this.#db.free(); } catch {} }
        if (this.#worker) this.#worker.terminate();
        this.#db = null;
        this.#worker = null;
        this.#initialized = false;
        this.#modelReady = false;
        this.#log('info', 'Destroyed');
    }
}
