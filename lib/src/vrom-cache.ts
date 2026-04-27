import type {
    VromRegistry,
    VromRegistryEntry,
    VromManifest,
    DownloadProgress,
    StorageEstimate,
} from './types.js';

const OPFS_ROOT = 'vecdb-vroms';
const REGISTRY_TTL_MS = 3_600_000; // 1 hour

const DEFAULT_REGISTRY_URL =
    'https://huggingface.co/datasets/philipp-zettl/vrom-registry/resolve/main/registry.json';

/**
 * OPFS-backed cache and registry manager for vROM packages.
 *
 * Handles hub:// URI resolution, CDN downloads with progress, and persistent
 * local caching via the Origin Private File System (OPFS).
 *
 * Used internally by {@link AgentMemory}, but exported for advanced use cases
 * that need direct cache control.
 *
 * Storage layout in OPFS:
 * ```
 * vecdb-vroms/
 *   registry.json              ← Cached registry (1-hour TTL)
 *   {vrom-id}/
 *     manifest.json
 *     index.json               ← Serialized HNSW graph
 * ```
 */
export class VromCache {
    #registryUrl: string;
    #registry: VromRegistry | null = null;
    #rootDir: FileSystemDirectoryHandle | null = null;

    /**
     * Create a VromCache instance.
     *
     * @param registryUrl - URL to the vROM registry JSON.
     *   Defaults to the official registry on HF Hub.
     */
    constructor(registryUrl?: string) {
        this.#registryUrl = registryUrl || DEFAULT_REGISTRY_URL;
    }

    // ─── OPFS Helpers ─────────────────────────────────────────────────

    async #getRoot(): Promise<FileSystemDirectoryHandle> {
        if (!this.#rootDir) {
            const root = await navigator.storage.getDirectory();
            this.#rootDir = await root.getDirectoryHandle(OPFS_ROOT, { create: true });
        }
        return this.#rootDir;
    }

    async #getVromDir(vromId: string): Promise<FileSystemDirectoryHandle> {
        const root = await this.#getRoot();
        return await root.getDirectoryHandle(vromId, { create: true });
    }

    async #readFile(dir: FileSystemDirectoryHandle, name: string): Promise<string | null> {
        try {
            const fh = await dir.getFileHandle(name, { create: false });
            const file = await fh.getFile();
            return await file.text();
        } catch (e: any) {
            if (e.name === 'NotFoundError') return null;
            throw e;
        }
    }

    async #writeFile(dir: FileSystemDirectoryHandle, name: string, content: string): Promise<void> {
        const fh = await dir.getFileHandle(name, { create: true });
        const w = await fh.createWritable();
        await w.write(content);
        await w.close();
    }

    async #fileExists(dir: FileSystemDirectoryHandle, name: string): Promise<boolean> {
        try {
            await dir.getFileHandle(name, { create: false });
            return true;
        } catch {
            return false;
        }
    }

    async #fileModTime(dir: FileSystemDirectoryHandle, name: string): Promise<number> {
        try {
            const fh = await dir.getFileHandle(name, { create: false });
            const file = await fh.getFile();
            return file.lastModified;
        } catch {
            return 0;
        }
    }

    // ─── Registry ─────────────────────────────────────────────────────

    /**
     * Fetch the vROM registry.
     *
     * Uses a three-tier lookup: in-memory cache → OPFS cache (1-hour TTL) → CDN fetch.
     * The registry is written to OPFS on a best-effort basis after fetching.
     *
     * @returns The full registry object containing all available vROM entries
     * @throws If the CDN fetch fails and no cached copy exists
     */
    async getRegistry(): Promise<VromRegistry> {
        if (this.#registry) return this.#registry;

        const root = await this.#getRoot();

        // Check OPFS cache freshness
        const modTime = await this.#fileModTime(root, 'registry.json');
        if (modTime && Date.now() - modTime < REGISTRY_TTL_MS) {
            const cached = await this.#readFile(root, 'registry.json');
            if (cached) {
                this.#registry = JSON.parse(cached);
                return this.#registry!;
            }
        }

        // Fetch from CDN
        const resp = await fetch(this.#registryUrl);
        if (!resp.ok) throw new Error(`Registry fetch failed: ${resp.status}`);
        const text = await resp.text();
        this.#registry = JSON.parse(text);

        try {
            await this.#writeFile(root, 'registry.json', text);
        } catch { /* best-effort cache */ }

        return this.#registry!;
    }

    /**
     * Resolve a vROM identifier to its registry entry.
     *
     * Supports bare IDs (`'hf-transformers-docs'`) and `hub://` URIs
     * (`'hub://hf-transformers-docs'`).
     *
     * @param vromIdOrUri - vROM identifier or hub:// URI
     * @returns The registry entry, or `null` if not found
     */
    async resolve(vromIdOrUri: string): Promise<VromRegistryEntry | null> {
        const id = vromIdOrUri.replace(/^hub:\/\//, '');
        const registry = await this.getRegistry();
        return registry.vroms.find(v => v.id === id) ?? null;
    }

    /**
     * List all available vROMs from the registry.
     *
     * @returns Array of all vROM entries
     */
    async list(): Promise<VromRegistryEntry[]> {
        const registry = await this.getRegistry();
        return registry.vroms;
    }

    // ─── Cache Ops ────────────────────────────────────────────────────

    /**
     * Check whether a vROM is cached locally in OPFS.
     *
     * @param vromId - vROM identifier
     * @returns `true` if the index file exists in OPFS
     */
    async isCached(vromId: string): Promise<boolean> {
        try {
            const dir = await this.#getVromDir(vromId);
            return await this.#fileExists(dir, 'index.json');
        } catch {
            return false;
        }
    }

    /**
     * Read the cached manifest for a vROM.
     *
     * @param vromId - vROM identifier
     * @returns Parsed manifest object, or `null` if not cached
     */
    async getCachedManifest(vromId: string): Promise<VromManifest | null> {
        try {
            const dir = await this.#getVromDir(vromId);
            const text = await this.#readFile(dir, 'manifest.json');
            return text ? JSON.parse(text) : null;
        } catch {
            return null;
        }
    }

    /**
     * Load a vROM's raw index JSON from OPFS cache.
     *
     * The returned string can be passed directly to `VectorDB.load()`.
     *
     * @param vromId - vROM identifier
     * @returns Raw JSON string, or `null` if not cached
     */
    async loadIndex(vromId: string): Promise<string | null> {
        const dir = await this.#getVromDir(vromId);
        return this.#readFile(dir, 'index.json');
    }

    /**
     * Download a vROM from CDN and write it to OPFS.
     *
     * Downloads the manifest first (small), then streams the index file (large)
     * with progress reporting via the `onProgress` callback.
     *
     * @param vromId - vROM identifier (used as the OPFS directory name)
     * @param entry - Registry entry containing CDN file URLs
     * @param onProgress - Optional progress callback. Receives `phase` ('manifest', 'index', 'done'),
     *   `file`, `loaded` bytes, and `total` bytes.
     *
     * @throws If manifest or index fetch fails (HTTP error)
     */
    async pull(
        vromId: string,
        entry: VromRegistryEntry,
        onProgress?: (p: DownloadProgress) => void,
    ): Promise<void> {
        const dir = await this.#getVromDir(vromId);

        // Manifest
        onProgress?.({ phase: 'manifest', file: 'manifest.json', loaded: 0, total: 1 });
        const mr = await fetch(entry.files.manifest);
        if (!mr.ok) throw new Error(`Manifest fetch failed: ${mr.status}`);
        await this.#writeFile(dir, 'manifest.json', await mr.text());
        onProgress?.({ phase: 'manifest', file: 'manifest.json', loaded: 1, total: 1 });

        // Index (streamed with progress)
        onProgress?.({ phase: 'index', file: 'index.json', loaded: 0, total: 0 });
        const ir = await fetch(entry.files.index);
        if (!ir.ok) throw new Error(`Index fetch failed: ${ir.status}`);

        const total = parseInt(ir.headers.get('content-length') || '0');
        const reader = ir.body!.getReader();
        const chunks: Uint8Array[] = [];
        let loaded = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            onProgress?.({ phase: 'index', file: 'index.json', loaded, total });
        }

        const buf = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            buf.set(chunk, offset);
            offset += chunk.length;
        }

        await this.#writeFile(dir, 'index.json', new TextDecoder().decode(buf));
        onProgress?.({ phase: 'done', file: 'index.json', loaded, total: loaded });
    }

    /**
     * Evict a vROM from the OPFS cache.
     *
     * Deletes all files (manifest + index) for the given vROM.
     * Safe to call even if the vROM is not cached (no-op).
     *
     * @param vromId - vROM identifier to evict
     */
    async evict(vromId: string): Promise<void> {
        try {
            const root = await this.#getRoot();
            await root.removeEntry(vromId, { recursive: true });
        } catch { /* noop */ }
    }

    /**
     * Get the browser's storage usage estimate for the current origin.
     *
     * @returns `used` bytes currently consumed and `quota` bytes available
     */
    async storageEstimate(): Promise<StorageEstimate> {
        const est = await navigator.storage.estimate();
        return { used: est.usage || 0, quota: est.quota || 0 };
    }
}
