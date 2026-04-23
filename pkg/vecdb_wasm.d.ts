/* tslint:disable */
/* eslint-disable */

/**
 * The main WASM-exposed vector database.
 */
export class VectorDB {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the vector dimensionality.
     */
    dim(): number;
    /**
     * Get metadata by ID.
     */
    get_metadata(id: number): string | undefined;
    /**
     * Get a vector by ID. Returns a Float32Array or null.
     */
    get_vector(id: number): Float32Array | undefined;
    /**
     * Insert a vector into the database. Returns the assigned ID.
     *
     * # Arguments
     * * `vector` - Float32Array of dimension `dim`
     * * `metadata` - Optional JSON string metadata
     */
    insert(vector: Float32Array, metadata?: string | null): number;
    /**
     * Batch insert multiple vectors.
     *
     * # Arguments
     * * `vectors` - Flat Float32Array of shape [n * dim]
     * * `n` - Number of vectors
     *
     * Returns the starting ID (IDs are sequential: start_id..start_id+n).
     */
    insert_batch(vectors: Float32Array, n: number): number;
    /**
     * Get the number of vectors in the database.
     */
    len(): number;
    /**
     * Load a database from a JSON string.
     */
    static load(json: string): VectorDB;
    /**
     * Create a new VectorDB.
     *
     * # Arguments
     * * `dim` - Vector dimensionality
     * * `metric` - Distance metric: "cosine", "euclidean", or "dot_product"
     * * `m` - Max connections per node per layer (default: 16)
     * * `ef_construction` - Construction search width (default: 128)
     * * `ef_search` - Default search width (default: 40)
     */
    constructor(dim: number, metric?: string | null, m?: number | null, ef_construction?: number | null, ef_search?: number | null);
    /**
     * Serialize the entire database to a JSON string (for persistence).
     */
    save(): string;
    /**
     * Search for k nearest neighbors.
     *
     * Returns a JSON string: [{"id": 0, "distance": 0.123, "metadata": "..."}, ...]
     */
    search(query: Float32Array, k: number): string;
    /**
     * Search with custom ef_search parameter (quality/speed tradeoff).
     */
    search_with_ef(query: Float32Array, k: number, ef_search: number): string;
    /**
     * Get index statistics as a JSON string.
     */
    stats(): string;
}

/**
 * Initialize panic hook for better error messages in browser console.
 */
export function init(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_vectordb_free: (a: number, b: number) => void;
    readonly init: () => void;
    readonly vectordb_dim: (a: number) => number;
    readonly vectordb_get_metadata: (a: number, b: number, c: number) => void;
    readonly vectordb_get_vector: (a: number, b: number, c: number) => void;
    readonly vectordb_insert: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly vectordb_insert_batch: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly vectordb_len: (a: number) => number;
    readonly vectordb_load: (a: number, b: number, c: number) => void;
    readonly vectordb_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly vectordb_save: (a: number, b: number) => void;
    readonly vectordb_search: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly vectordb_search_with_ef: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly vectordb_stats: (a: number, b: number) => void;
    readonly __wbindgen_export: (a: number) => void;
    readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export3: (a: number, b: number) => number;
    readonly __wbindgen_export4: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
