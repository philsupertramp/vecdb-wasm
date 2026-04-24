# Getting Started

## Installation

```bash
npm install vecdb-wasm
```

The package includes:
- ESM and CJS bundles with full TypeScript declarations
- A background Web Worker for ONNX embedding inference
- The compiled WASM binary (~172 KB) for HNSW vector search

### Browser Requirements

VecDB-WASM runs entirely in the browser. It requires:

| Feature | Used For | Support |
|---------|----------|---------|
| [WebAssembly](https://caniuse.com/wasm) | HNSW vector search engine | All modern browsers |
| [Web Workers](https://caniuse.com/webworkers) | Background embedding inference | All modern browsers |
| [OPFS](https://caniuse.com/native-filesystem-api) | Persistent vROM cache | Chrome 86+, Firefox 111+, Safari 15.2+ |
| [ES Modules in Workers](https://caniuse.com/mdn-api_worker_worker_ecmascript_modules) | Worker `type: 'module'` | Chrome 80+, Firefox 114+, Safari 15+ |

> **Note:** The embedding worker loads [transformers.js](https://huggingface.co/docs/transformers.js) from CDN at runtime. The first model load requires an internet connection; subsequent loads use the browser's Cache API.

## Quick Start

```typescript
import { AgentMemory } from 'vecdb-wasm';

// 1. Create and initialize
const memory = new AgentMemory();
await memory.init();

// 2. Mount a pre-built knowledge base
await memory.mount('hf-transformers-docs');

// 3. Search with natural language
const results = await memory.search('how to fine-tune a model');

// 4. Format for LLM context injection
const context = memory.formatContext(results, { maxTokens: 2000 });

// 5. Clean up when done
memory.destroy();
```

That's it. Five lines of meaningful code to go from zero to a searchable knowledge base with semantic search.

## What Happens Under the Hood

When you run the code above, VecDB-WASM:

1. **`init()`** — Loads the 172 KB WASM binary (HNSW engine) and spawns a background Web Worker for embedding inference.

2. **`mount('hf-transformers-docs')`** — Does four things:
   - Checks the [vROM Registry](https://huggingface.co/datasets/philipp-zettl/vrom-registry) for the requested knowledge base
   - Downloads the pre-computed HNSW index (~12 MB) from Hugging Face CDN, or loads it from the OPFS cache if already downloaded
   - Deserializes the index into the WASM engine via `VectorDB.load()`
   - Loads the required embedding model (`all-MiniLM-L6-v2`, ~22 MB q8) in the background worker, or skips this if the same model is already loaded

3. **`search('how to fine-tune a model')`** — Embeds the query text in the background worker (~50ms), then runs HNSW approximate nearest neighbor search in WASM (<1ms).

4. **`formatContext(results)`** — Concatenates result texts with source URLs into a string ready for LLM system/user prompt injection.

## 5-Minute Tutorial

### Step 1: Set Up a Project

```bash
mkdir my-rag-app && cd my-rag-app
npm init -y
npm install vecdb-wasm
```

### Step 2: Create the App

Create `index.html`:

```html
<!DOCTYPE html>
<html>
<head><title>My RAG App</title></head>
<body>
  <input id="query" placeholder="Ask anything about HF Transformers..." style="width: 400px">
  <button id="search">Search</button>
  <pre id="results"></pre>

  <script type="module">
    import { AgentMemory } from './node_modules/vecdb-wasm/dist/index.js';

    const memory = new AgentMemory({ logLevel: 'info' });

    // Show loading progress
    memory.onProgress(({ file, loaded, total }) => {
      const pct = total > 0 ? ((loaded / total) * 100).toFixed(0) : '?';
      document.getElementById('results').textContent = `Loading ${file}... ${pct}%`;
    });

    // Initialize
    await memory.init();
    document.getElementById('results').textContent = 'WASM ready. Mounting knowledge base...';

    // Mount with download progress
    const status = await memory.mount('hf-transformers-docs', {
      onProgress: ({ phase, loaded, total }) => {
        if (phase === 'index' && total > 0) {
          const mb = (loaded / 1e6).toFixed(1);
          document.getElementById('results').textContent = `Downloading index... ${mb} MB`;
        }
      }
    });

    document.getElementById('results').textContent =
      `Ready! ${status.vectors} vectors, ${status.dim}d, model: ${status.model}`;

    // Search handler
    document.getElementById('search').addEventListener('click', async () => {
      const query = document.getElementById('query').value;
      if (!query) return;

      const results = await memory.search(query, {
        topK: 5,
        expandContext: true,
        contextWindow: 1,
      });

      let output = '';
      for (const r of results) {
        output += `[d=${r.distance.toFixed(4)}] ${r.metadata.section_heading}\n`;
        output += `${r.text.slice(0, 200)}...\n`;
        if (r.metadata.url) output += `Source: ${r.metadata.url}\n`;
        output += '\n---\n\n';
      }
      document.getElementById('results').textContent = output;
    });
  </script>
</body>
</html>
```

### Step 3: Serve and Test

```bash
npx serve .
# Open http://localhost:3000
```

Type a query like *"how to use the pipeline API"* and hit Search. Results appear in milliseconds.

### Step 4: Use the Context in an LLM Call

```typescript
const results = await memory.search(userQuestion, { topK: 5, expandContext: true });
const context = memory.formatContext(results, { maxTokens: 2000 });

// Inject into any LLM API
const response = await fetch('https://api.openai.com/v1/chat/completions', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: `Answer using this context:\n\n${context}` },
      { role: 'user', content: userQuestion },
    ],
  }),
});
```

## Framework Integration

### Vite

VecDB-WASM works out of the box with Vite. Import normally:

```typescript
import { AgentMemory } from 'vecdb-wasm';
```

The worker and WASM paths auto-resolve via `import.meta.url`. If you encounter issues with the worker path, override it:

```typescript
import workerUrl from 'vecdb-wasm/embed-worker?url';

const memory = new AgentMemory({
  workerPath: workerUrl,
});
```

### Next.js (App Router)

The SDK is browser-only. Use dynamic imports to avoid SSR:

```typescript
'use client';

import { useEffect, useRef, useState } from 'react';
import type { AgentMemory as AgentMemoryType, MountStatus } from 'vecdb-wasm';

export function useAgentMemory(vromId: string) {
  const memoryRef = useRef<AgentMemoryType | null>(null);
  const [status, setStatus] = useState<MountStatus | null>(null);

  useEffect(() => {
    let destroyed = false;

    (async () => {
      const { AgentMemory } = await import('vecdb-wasm');
      if (destroyed) return;

      const memory = new AgentMemory({ logLevel: 'warn' });
      await memory.init();
      const mountStatus = await memory.mount(vromId);

      if (destroyed) { memory.destroy(); return; }
      memoryRef.current = memory;
      setStatus(mountStatus);
    })();

    return () => {
      destroyed = true;
      memoryRef.current?.destroy();
    };
  }, [vromId]);

  return { memory: memoryRef.current, status };
}
```

### Vanilla (CDN / No Bundler)

```html
<script type="module">
  import { AgentMemory } from 'https://cdn.jsdelivr.net/npm/vecdb-wasm/dist/index.js';

  const memory = new AgentMemory({
    // Explicit paths when not using a bundler
    workerPath: 'https://cdn.jsdelivr.net/npm/vecdb-wasm/dist/embed-worker.js',
    wasmPkgPath: 'https://cdn.jsdelivr.net/npm/vecdb-wasm/wasm-pkg/vecdb_wasm.js',
  });

  await memory.init();
  await memory.mount('hf-transformers-docs');
</script>
```

## Next Steps

- **[API Reference](./api-reference.md)** — Full documentation of every class, method, option, and type
- **[Guides](./guides.md)** — Deep dives on vROMs, context expansion, custom knowledge bases, and the Python CLI
- **[Architecture](./architecture.md)** — How the HNSW engine, worker protocol, and OPFS cache work internally
