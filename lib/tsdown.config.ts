import { defineConfig } from 'tsdown';
import { copyFileSync, mkdirSync, readdirSync, renameSync, existsSync } from 'node:fs';

function stabilizeDtsNames() {
    // tsdown generates hashed .d.ts/.d.cts names.
    // Rename them to stable names so package.json exports map works.
    const dir = 'dist';
    
    // Safety check in case the directory doesn't exist yet
    if (!existsSync(dir)) return;

    for (const file of readdirSync(dir)) {
        if (file.startsWith('index-') && file.endsWith('.d.ts')) {
            renameSync(`${dir}/${file}`, `${dir}/index.d.ts`);
        } else if (file.startsWith('index-') && file.endsWith('.d.ts.map')) {
            renameSync(`${dir}/${file}`, `${dir}/index.d.ts.map`);
        } else if (file.startsWith('index-') && file.endsWith('.d.cts')) {
            renameSync(`${dir}/${file}`, `${dir}/index.d.cts`);
        } else if (file.startsWith('index-') && file.endsWith('.d.cts.map')) {
            renameSync(`${dir}/${file}`, `${dir}/index.d.cts.map`);
        } else if (file.startsWith('embed-worker-') && file.endsWith('.d.ts')) {
            renameSync(`${dir}/${file}`, `${dir}/embed-worker.d.ts`);
        } else if (file.startsWith('embed-worker-') && file.endsWith('.d.ts.map')) {
            renameSync(`${dir}/${file}`, `${dir}/embed-worker.d.ts.map`);
        }
    }
}

export default [
    // ── Main SDK: ESM + CJS + declarations ──
    defineConfig({
        entry: { index: 'src/index.ts' },
        format: ['esm', 'cjs'],
        outDir: 'dist',
        dts: true,
        target: 'es2022',
        platform: 'browser',
        clean: true,
        onSuccess() {
            // 1. Copy WASM binary
            mkdirSync('dist', { recursive: true });
            try {
                copyFileSync('wasm-pkg/vrom_js_bg.wasm', 'dist/vrom_js_bg.wasm');
            } catch {}

            // 2. Stabilize .d.ts filenames (remove content hashes)
            stabilizeDtsNames();
        },
    }),

    // ── Embed Worker: separate file, ESM only + declarations ──
    defineConfig({
        entry: { 'embed-worker': 'src/embed-worker.ts' },
        format: ['esm'],
        outDir: 'dist',
        dts: true,
        target: 'es2022',
        platform: 'browser',
        clean: false,
        onSuccess() {
            stabilizeDtsNames();
        },
    }),
];
