/**
 * sdk/embed-worker.js — Background Web Worker for transformers.js embedding.
 *
 * Runs ONNX inference off the main thread. Supports model diffing —
 * if the requested model matches the already-loaded one, skips reload.
 *
 * Protocol (all messages tagged with source: 'vecdb'):
 *   Main → Worker:
 *     { type: 'load', modelId, dtype }           — Load or diff model
 *     { type: 'embed', texts, id }               — Embed a batch
 *     { type: 'unload' }                         — Release model
 *     { type: 'get-model' }                      — Query current model
 *
 *   Worker → Main:
 *     { status: 'dl-progress', file, loaded, total }
 *     { status: 'ready', dim, modelId, dtype, cached }  — Model ready
 *     { status: 'result', id, embeddings, dims }
 *     { status: 'error', id?, message }
 *     { status: 'unloaded' }
 *     { status: 'model-info', modelId, dtype, dim }
 */

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

const MSG_TAG = 'vecdb';

let extractor = null;
let currentModel = null;
let currentDtype = null;
let currentDim = null;
let loadingPromise = null;

function send(msg) {
    self.postMessage({ ...msg, source: MSG_TAG });
}

function sendTransfer(msg, transfer) {
    self.postMessage({ ...msg, source: MSG_TAG }, transfer);
}

self.addEventListener('message', async (event) => {
    const { type, id } = event.data;

    try {
        switch (type) {
            case 'load': {
                const { modelId, dtype } = event.data;

                // ── Model Diffing ──
                // If the requested model is already loaded, skip the expensive reload
                if (extractor && currentModel === modelId && currentDtype === dtype && currentDim) {
                    send({ status: 'ready', dim: currentDim, modelId, dtype, cached: true });
                    break;
                }

                // Different model requested — dispose old one
                if (extractor) {
                    try { await extractor.dispose(); } catch {}
                    extractor = null;
                }

                currentModel = modelId;
                currentDtype = dtype;
                currentDim = null;

                extractor = await pipeline('feature-extraction', modelId, {
                    dtype,
                    progress_callback: (p) => {
                        if (p.status === 'progress' && p.total > 0) {
                            send({ status: 'dl-progress', file: p.file, loaded: p.loaded, total: p.total });
                        }
                    },
                });

                // Probe dimension
                const test = await extractor(['test'], { pooling: 'mean', normalize: true });
                currentDim = test.dims[1];

                send({ status: 'ready', dim: currentDim, modelId, dtype, cached: false });
                break;
            }

            case 'embed': {
                if (!extractor) {
                    send({ status: 'error', id, message: 'Model not loaded' });
                    return;
                }

                const { texts } = event.data;
                const output = await extractor(texts, { pooling: 'mean', normalize: true });

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
                currentDim = null;
                loadingPromise = null;
                send({ status: 'unloaded' });
                break;
            }

            case 'get-model': {
                send({
                    status: 'model-info',
                    modelId: currentModel,
                    dtype: currentDtype,
                    dim: currentDim,
                    loaded: !!extractor,
                });
                break;
            }
        }
    } catch (err) {
        send({ status: 'error', id, message: err.message || String(err) });
    }
});
