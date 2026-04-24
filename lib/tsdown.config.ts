import { defineConfig } from 'tsdown';
import { copyFileSync, mkdirSync, readdirSync, renameSync } from 'node:fs';

function stabilizeDtsNames() {
    // tsdown generates hashed .d.ts/.d.cts names (e.g. index-DREOnhJl.d.ts).
    // Rename them to stable names so package.json exports map works.
    const dir = 'dist';
    for (const file of readdirSync(dir)) {
        if (file.match(/^index-[A-Za-z0-9_]+\.d\.ts$/)) {
            renameSync(`${dir}/${file}`, `${dir}/index.d.ts`);
        } else if (file.match(/^index-[A-Za-z0-9_]+\.d\.ts\.map$/)) {
            renameSync(`${dir}/${file}`, `${dir}/index.d.ts.map`);
        } else if (file.match(/^index-[A-Za-z0-9_]+\.d\.cts$/)) {
            renameSync(`${dir}/${file}`, `${dir}/index.d.cts`);
        } else if (file.match(/^index-[A-Za-z0-9_]+\.d\.cts\.map$/)) {
            renameSync(`${dir}/${file}`, `${dir}/index.d.cts.map`);
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
                copyFileSync('wasm-pkg/vecdb_wasm_bg.wasm', 'dist/vecdb_wasm_bg.wasm');
            } catch {}

            // 2. Stabilize .d.ts filenames (remove content hashes)
            stabilizeDtsNames();
        },
    }),

    // ── Embed Worker: separate file, ESM only ──
    defineConfig({
        entry: { 'embed-worker': 'src/embed-worker.ts' },
        format: ['esm'],
        outDir: 'dist',
        dts: false,
        target: 'es2022',
        platform: 'browser',
        clean: false,
    }),
];
