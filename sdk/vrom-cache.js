/**
 * vrom-cache.js — OPFS-backed cache for vROM packages with hub:// URI resolution.
 *
 * Handles:
 *   - hub:// URI resolution → CDN URLs
 *   - OPFS read/write for offline-first access
 *   - Cache-hit detection to skip network fetches
 *   - Registry fetching and caching
 *
 * Storage layout in OPFS:
 *   vecdb-vroms/
 *     registry.json          ← Cached registry (TTL: 1 hour)
 *     {vrom-id}/
 *       manifest.json
 *       index.json
 */

const OPFS_ROOT = 'vecdb-vroms';
const REGISTRY_TTL_MS = 3600_000; // 1 hour

// Default CDN base — can be overridden
const DEFAULT_REGISTRY_URL = 'https://huggingface.co/datasets/philipp-zettl/vrom-registry/resolve/main/registry.json';

export class VromCache {
    /** @type {string} */ #registryUrl;
    /** @type {object|null} */ #registry = null;
    /** @type {FileSystemDirectoryHandle|null} */ #rootDir = null;

    constructor(registryUrl = DEFAULT_REGISTRY_URL) {
        this.#registryUrl = registryUrl;
    }

    // ─── OPFS Helpers ─────────────────────────────────────────────────

    async #getRoot() {
        if (!this.#rootDir) {
            const root = await navigator.storage.getDirectory();
            this.#rootDir = await root.getDirectoryHandle(OPFS_ROOT, { create: true });
        }
        return this.#rootDir;
    }

    async #getVromDir(vromId) {
        const root = await this.#getRoot();
        return root.getDirectoryHandle(vromId, { create: true });
    }

    async #readFile(dirHandle, filename) {
        try {
            const fh = await dirHandle.getFileHandle(filename, { create: false });
            const file = await fh.getFile();
            return await file.text();
        } catch (e) {
            if (e.name === 'NotFoundError') return null;
            throw e;
        }
    }

    async #writeFile(dirHandle, filename, content) {
        const fh = await dirHandle.getFileHandle(filename, { create: true });
        const w = await fh.createWritable();
        await w.write(content);
        await w.close();
    }

    async #fileExists(dirHandle, filename) {
        try {
            await dirHandle.getFileHandle(filename, { create: false });
            return true;
        } catch { return false; }
    }

    async #fileModTime(dirHandle, filename) {
        try {
            const fh = await dirHandle.getFileHandle(filename, { create: false });
            const file = await fh.getFile();
            return file.lastModified;
        } catch { return 0; }
    }

    // ─── Registry ─────────────────────────────────────────────────────

    /**
     * Get the vROM registry. Uses OPFS cache with TTL, falls back to CDN.
     * @returns {Promise<object>} The registry object with .vroms array
     */
    async getRegistry() {
        if (this.#registry) return this.#registry;

        const root = await this.#getRoot();

        // Check OPFS cache freshness
        const modTime = await this.#fileModTime(root, 'registry.json');
        if (modTime && (Date.now() - modTime) < REGISTRY_TTL_MS) {
            const cached = await this.#readFile(root, 'registry.json');
            if (cached) {
                this.#registry = JSON.parse(cached);
                return this.#registry;
            }
        }

        // Fetch from CDN
        const resp = await fetch(this.#registryUrl);
        if (!resp.ok) throw new Error(`Registry fetch failed: ${resp.status}`);
        const text = await resp.text();
        this.#registry = JSON.parse(text);

        // Cache to OPFS
        try { await this.#writeFile(root, 'registry.json', text); } catch {}

        return this.#registry;
    }

    /**
     * Resolve a vROM identifier to its registry entry.
     * Supports: 'hf-transformers-docs' or 'hub://hf-transformers-docs'
     * @param {string} vromIdOrUri
     * @returns {Promise<object|null>} Registry entry or null
     */
    async resolve(vromIdOrUri) {
        const id = vromIdOrUri.replace(/^hub:\/\//, '');
        const registry = await this.getRegistry();
        return registry.vroms.find(v => v.id === id) || null;
    }

    /**
     * List all available vROMs from registry.
     * @returns {Promise<object[]>}
     */
    async list() {
        const registry = await this.getRegistry();
        return registry.vroms;
    }

    // ─── Cache Operations ──────────────────────────────────────────────

    /**
     * Check if a vROM is cached locally in OPFS.
     * @param {string} vromId
     * @returns {Promise<boolean>}
     */
    async isCached(vromId) {
        try {
            const dir = await this.#getVromDir(vromId);
            return await this.#fileExists(dir, 'index.json');
        } catch { return false; }
    }

    /**
     * Get the cached manifest for a vROM (null if not cached).
     * @param {string} vromId
     * @returns {Promise<object|null>}
     */
    async getCachedManifest(vromId) {
        try {
            const dir = await this.#getVromDir(vromId);
            const text = await this.#readFile(dir, 'manifest.json');
            return text ? JSON.parse(text) : null;
        } catch { return null; }
    }

    /**
     * Load a vROM's index JSON from OPFS cache.
     * @param {string} vromId
     * @returns {Promise<string|null>} Raw JSON string for VectorDB.load()
     */
    async loadIndex(vromId) {
        const dir = await this.#getVromDir(vromId);
        return await this.#readFile(dir, 'index.json');
    }

    /**
     * Fetch a vROM from CDN and cache it to OPFS.
     * @param {string} vromId
     * @param {object} registryEntry - The registry entry for this vROM
     * @param {function} [onProgress] - Progress callback ({phase, loaded, total, file})
     * @returns {Promise<void>}
     */
    async pull(vromId, registryEntry, onProgress) {
        const dir = await this.#getVromDir(vromId);
        const files = registryEntry.files;

        // Fetch manifest
        onProgress?.({ phase: 'manifest', file: 'manifest.json', loaded: 0, total: 1 });
        const manifestResp = await fetch(files.manifest);
        if (!manifestResp.ok) throw new Error(`Manifest fetch failed: ${manifestResp.status}`);
        const manifestText = await manifestResp.text();
        await this.#writeFile(dir, 'manifest.json', manifestText);
        onProgress?.({ phase: 'manifest', file: 'manifest.json', loaded: 1, total: 1 });

        // Fetch index (the large file — stream with progress)
        onProgress?.({ phase: 'index', file: 'index.json', loaded: 0, total: 0 });
        const indexResp = await fetch(files.index);
        if (!indexResp.ok) throw new Error(`Index fetch failed: ${indexResp.status}`);

        const total = parseInt(indexResp.headers.get('content-length') || '0');
        const reader = indexResp.body.getReader();
        const chunks = [];
        let loaded = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            onProgress?.({ phase: 'index', file: 'index.json', loaded, total });
        }

        // Combine chunks and write
        const indexBuffer = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            indexBuffer.set(chunk, offset);
            offset += chunk.length;
        }
        const indexText = new TextDecoder().decode(indexBuffer);
        await this.#writeFile(dir, 'index.json', indexText);

        onProgress?.({ phase: 'done', file: 'index.json', loaded, total: loaded });
    }

    /**
     * Evict a vROM from OPFS cache.
     * @param {string} vromId
     */
    async evict(vromId) {
        try {
            const root = await this.#getRoot();
            await root.removeEntry(vromId, { recursive: true });
        } catch {}
    }

    /**
     * Get total OPFS usage estimate.
     * @returns {Promise<{used: number, quota: number}>}
     */
    async storageEstimate() {
        const est = await navigator.storage.estimate();
        return { used: est.usage || 0, quota: est.quota || 0 };
    }
}
