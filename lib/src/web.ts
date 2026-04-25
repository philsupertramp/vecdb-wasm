import initWasm, * as wasm from '../wasm-web/vrom_js.js';
import { AgentMemoryCore } from './agent-memory.js';
import type { AgentMemoryOptions } from './types.js';
export { VromCache } from './vrom-cache.js';

export class AgentMemory extends AgentMemoryCore {
    constructor(options?: AgentMemoryOptions) {
        super(async () => {
            await initWasm(options?.wasmPkgPath); 
            return wasm;
        }, options);
    }
}

export type {
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
} from './types.js';

