// ─── VectorDB WASM Bindings (from wasm-bindgen) ──────────────────────────────

/** The WASM-exposed VectorDB class. */
export interface VectorDB {
    free(): void;
    dim(): number;
    len(): number;
    stats(): string;
    save(): string;
    insert(vector: Float32Array, metadata?: string | null): number;
    insert_batch(vectors: Float32Array, n: number): number;
    search(query: Float32Array, k: number): string;
    search_with_ef(query: Float32Array, k: number, ef_search: number): string;
    get_vector(id: number): Float32Array | undefined;
    get_metadata(id: number): string | undefined;
}

export interface VectorDBConstructor {
    new (dim: number, metric?: string | null, m?: number | null,
         ef_construction?: number | null, ef_search?: number | null): VectorDB;
    load(json: string): VectorDB;
}

export interface WasmModule {
    default: (input?: any) => Promise<any>;
    VectorDB: VectorDBConstructor;
}

// ─── Worker Protocol ──────────────────────────────────────────────────────────

export interface WorkerLoadMessage {
    type: 'load';
    modelId: string;
    dtype: string;
}

export interface WorkerEmbedMessage {
    type: 'embed';
    texts: string[];
    id: string;
}

export interface WorkerUnloadMessage {
    type: 'unload';
}

export interface WorkerGetModelMessage {
    type: 'get-model';
}

export type WorkerInMessage =
    | WorkerLoadMessage
    | WorkerEmbedMessage
    | WorkerUnloadMessage
    | WorkerGetModelMessage;

export interface WorkerReadyMessage {
    source: 'vecdb';
    status: 'ready';
    dim: number;
    modelId: string;
    dtype: string;
    cached: boolean;
}

export interface WorkerResultMessage {
    source: 'vecdb';
    status: 'result';
    id: string;
    embeddings: Float32Array;
    dims: number[];
}

export interface WorkerProgressMessage {
    source: 'vecdb';
    status: 'dl-progress';
    file: string;
    loaded: number;
    total: number;
}

export interface WorkerErrorMessage {
    source: 'vecdb';
    status: 'error';
    id?: string;
    message: string;
}

export interface WorkerUnloadedMessage {
    source: 'vecdb';
    status: 'unloaded';
}

export interface WorkerModelInfoMessage {
    source: 'vecdb';
    status: 'model-info';
    modelId: string | null;
    dtype: string | null;
    dim: number | null;
    loaded: boolean;
}

export type WorkerOutMessage =
    | WorkerReadyMessage
    | WorkerResultMessage
    | WorkerProgressMessage
    | WorkerErrorMessage
    | WorkerUnloadedMessage
    | WorkerModelInfoMessage;

// ─── SDK Public Types ─────────────────────────────────────────────────────────

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

export interface MountOptions {
    /** Progress callback for CDN download. */
    onProgress?: (progress: DownloadProgress) => void;
    /** Force re-download even if cached in OPFS. Default: false */
    forceDownload?: boolean;
}

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

export interface FormatContextOptions {
    /** Include source URLs in output. Default: true */
    includeSources?: boolean;
    /** Approximate token budget (stops adding chunks when exceeded). */
    maxTokens?: number;
}

export interface MountStatus {
    /** Currently mounted vROM ID, or null */
    activeVrom: string | null;
    /** vROM version string */
    version: string | null;
    /** Whether the vROM is searchable (index loaded + model ready) */
    ready: boolean;
    /** Number of vectors in the mounted index */
    vectors: number;
    /** Vector dimensionality */
    dim: number;
    /** Currently loaded embedding model ID */
    model: string | null;
}

export interface SearchResult {
    /** The chunk text (expanded if expandContext was true) */
    text: string;
    /** Full chunk metadata */
    metadata: ChunkMetadata & Record<string, any>;
    /** Cosine distance (lower = more similar) */
    distance: number;
    /** Vector ID in the HNSW index */
    id: number;
}

export interface ChunkMetadata {
    chunk_id: number;
    text: string;
    source_file: string;
    section_heading: string;
    prev_chunk_id: number | null;
    next_chunk_id: number | null;
    url: string;
    doc_title: string;
    /** Set by context expansion */
    _expanded?: boolean;
    /** Number of chunks in expanded context */
    _contextChunks?: number;
}

export interface DownloadProgress {
    phase: 'manifest' | 'index' | 'done';
    file: string;
    loaded: number;
    total: number;
}

export interface VromRegistryEntry {
    id: string;
    name: string;
    description: string;
    version: string;
    vectors: number;
    dimensions: number;
    tokens: number;
    size_mb: number;
    model: string;
    tags: string[];
    official: boolean;
    files: {
        manifest: string;
        index: string;
        chunks: string;
    };
}

export interface VromRegistry {
    version: string;
    description: string;
    base_url: string;
    vroms: VromRegistryEntry[];
}

export interface VromManifest {
    vrom_id: string;
    version: string;
    description: string;
    source: string;
    embedding_spec: {
        model: string;
        model_source?: string;
        dimensions: number;
        quantization: string;
        distance_metric: string;
        normalized: boolean;
    };
    hnsw_config: Record<string, any>;
    vector_count: number;
    total_tokens: number;
    total_chunks: number;
    corpus_hash: string;
    created_at: string;
    chunk_strategy: Record<string, any>;
}

export interface StorageEstimate {
    used: number;
    quota: number;
}
