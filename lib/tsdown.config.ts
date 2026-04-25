import { defineConfig } from 'tsdown';
import { readdirSync, renameSync, existsSync } from 'node:fs';

function stabilizeDtsNames() {
    const dir = 'dist';
    if (!existsSync(dir)) return;

    for (const file of readdirSync(dir)) {
        // This regex looks for index-HASH, web-HASH, or embed-worker-HASH 
        // and safely strips the hash out of the filename.
        const match = file.match(/^(index|web|embed-worker)-[a-zA-Z0-9_-]+\.(d\.c?ts(?:\.map)?)$/);
        
        if (match) {
            const [, name, ext] = match; // name = 'web', ext = 'd.ts'
            renameSync(`${dir}/${file}`, `${dir}/${name}.${ext}`);
        }
    }
}

export default [
    // ── Main SDK: ESM + CJS + declarations ──
    defineConfig({
        entry: [
          'src/index.ts', 
          'src/web.ts', 
          'src/embed-worker.ts'
        ],
        format: ['esm', 'cjs'], 
        outDir: 'dist',
        
        external: [
          /wasm-bundler/, 
          /wasm-web/
        ],
        dts: true,
        target: 'es2022',
        platform: 'browser',
        clean: true,
        onSuccess() {
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
        clean: false, // Important: don't clean the previous step's output!
        onSuccess() {
            stabilizeDtsNames();
        },
    }),
];
