/**
 * embed-worker.js — Background Web Worker for transformers.js embedding
 *
 * Runs the ONNX embedding model off the main thread so the UI stays responsive.
 * Uses a singleton pattern to ensure the model loads only once.
 *
 * Protocol:
 *   Main → Worker:
 *     { type: 'load', modelId, dtype }        — Load/switch model
 *     { type: 'embed', texts, id }            — Embed a batch of texts
 *     { type: 'unload' }                      — Release model memory
 *
 *   Worker → Main:
 *     { status: 'initiate', file, name }      — Download starting
 *     { status: 'progress', file, progress, loaded, total }
 *     { status: 'done', file }                — File cached
 *     { status: 'ready', dim }                — Model ready, reports dimension
 *     { status: 'result', id, embeddings, dims }  — Embedding result (transferable)
 *     { status: 'error', id?, message }       — Error
 *     { status: 'unloaded' }                  — Model released
 */

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

let extractor = null;
let currentModel = null;
let currentDtype = null;
let loadingPromise = null;

async function loadModel(modelId, dtype, progressCallback) {
    // If already loading the same model, wait for it
    if (loadingPromise && currentModel === modelId && currentDtype === dtype) {
        return loadingPromise;
    }

    // If switching models, dispose old one
    if (extractor) {
        try { await extractor.dispose(); } catch {}
        extractor = null;
    }

    currentModel = modelId;
    currentDtype = dtype;

    loadingPromise = pipeline('feature-extraction', modelId, {
        dtype: dtype,
        progress_callback: progressCallback,
    });

    extractor = await loadingPromise;
    return extractor;
}

self.addEventListener('message', async (event) => {
    const { type, id } = event.data;

    try {
        switch (type) {
            case 'load': {
                const { modelId, dtype } = event.data;
                await loadModel(modelId, dtype, (p) => {
                    // Relay all progress events to main thread
                    self.postMessage(p);
                });

                // Probe dimension by embedding a test string
                const test = await extractor(['test'], { pooling: 'mean', normalize: true });
                const dim = test.dims[1];

                self.postMessage({ status: 'ready', dim });
                break;
            }

            case 'embed': {
                if (!extractor) {
                    self.postMessage({ status: 'error', id, message: 'Model not loaded' });
                    return;
                }

                const { texts } = event.data;
                const output = await extractor(texts, { pooling: 'mean', normalize: true });

                // Copy to a fresh Float32Array for zero-copy transfer
                const float32 = new Float32Array(output.data);
                self.postMessage(
                    { status: 'result', id, embeddings: float32, dims: output.dims },
                    [float32.buffer]  // Transfer the buffer, not copy
                );
                break;
            }

            case 'unload': {
                if (extractor) {
                    try { await extractor.dispose(); } catch {}
                    extractor = null;
                }
                currentModel = null;
                currentDtype = null;
                loadingPromise = null;
                self.postMessage({ status: 'unloaded' });
                break;
            }
        }
    } catch (err) {
        self.postMessage({ status: 'error', id, message: err.message || String(err) });
    }
});
