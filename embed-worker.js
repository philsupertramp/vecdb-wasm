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
 *   Worker → Main (all tagged with source: 'vecdb'):
 *     { source, status: 'progress', file, progress, loaded, total }
 *     { source, status: 'ready', dim }                — Model ready
 *     { source, status: 'result', id, embeddings, dims }
 *     { source, status: 'error', id?, message }
 *     { source, status: 'unloaded' }
 */

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

const MSG_TAG = 'vecdb';

let extractor = null;
let currentModel = null;
let currentDtype = null;
let loadingPromise = null;

function send(msg) {
    self.postMessage({ ...msg, source: MSG_TAG });
}

function sendTransfer(msg, transfer) {
    self.postMessage({ ...msg, source: MSG_TAG }, transfer);
}

async function loadModel(modelId, dtype) {
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
        progress_callback: (p) => {
            // Only relay download progress events we care about.
            // Transformers.js emits: initiate, download, progress, done, ready
            // We only forward 'progress' (with loaded/total) so the main thread
            // can update the progress bar. Everything else is ignored to avoid
            // status collisions (e.g. transformers.js 'ready' vs our 'ready').
            if (p.status === 'progress' && p.total > 0) {
                send({ status: 'dl-progress', file: p.file, loaded: p.loaded, total: p.total });
            }
        },
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
                await loadModel(modelId, dtype);

                // Probe dimension by embedding a test string
                const test = await extractor(['test'], { pooling: 'mean', normalize: true });
                const dim = test.dims[1];

                send({ status: 'ready', dim });
                break;
            }

            case 'embed': {
                if (!extractor) {
                    send({ status: 'error', id, message: 'Model not loaded' });
                    return;
                }

                const { texts } = event.data;
                const output = await extractor(texts, { pooling: 'mean', normalize: true });

                // Copy to a fresh Float32Array for zero-copy transfer
                const float32 = new Float32Array(output.data);
                sendTransfer(
                    { status: 'result', id, embeddings: float32, dims: output.dims },
                    [float32.buffer]
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
                send({ status: 'unloaded' });
                break;
            }
        }
    } catch (err) {
        send({ status: 'error', id, message: err.message || String(err) });
    }
});
