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

export class VromCache {
    #registryUrl: string;
    #registry: VromRegistry | null = null;
    #rootDir: FileSystemDirectoryHandle | null = null;

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
        return root.getDirectoryHandle(vromId, { create: true });
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

    async resolve(vromIdOrUri: string): Promise<VromRegistryEntry | null> {
        const id = vromIdOrUri.replace(/^hub:\/\//, '');
        const registry = await this.getRegistry();
        return registry.vroms.find(v => v.id === id) ?? null;
    }

    async list(): Promise<VromRegistryEntry[]> {
        const registry = await this.getRegistry();
        return registry.vroms;
    }

    // ─── Cache Ops ────────────────────────────────────────────────────

    async isCached(vromId: string): Promise<boolean> {
        try {
            const dir = await this.#getVromDir(vromId);
            return await this.#fileExists(dir, 'index.json');
        } catch {
            return false;
        }
    }

    async getCachedManifest(vromId: string): Promise<VromManifest | null> {
        try {
            const dir = await this.#getVromDir(vromId);
            const text = await this.#readFile(dir, 'manifest.json');
            return text ? JSON.parse(text) : null;
        } catch {
            return null;
        }
    }

    async loadIndex(vromId: string): Promise<string | null> {
        const dir = await this.#getVromDir(vromId);
        return this.#readFile(dir, 'index.json');
    }

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

    async evict(vromId: string): Promise<void> {
        try {
            const root = await this.#getRoot();
            await root.removeEntry(vromId, { recursive: true });
        } catch { /* noop */ }
    }

    async storageEstimate(): Promise<StorageEstimate> {
        const est = await navigator.storage.estimate();
        return { used: est.usage || 0, quota: est.quota || 0 };
    }
}
