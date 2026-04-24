/**
 * embed-worker.ts — Background Web Worker for transformers.js ONNX embedding.
 *
 * Supports model diffing: if the requested model is already loaded, skips reload.
 * All outgoing messages are tagged with `source: 'vecdb'` to avoid collisions
 * with transformers.js internal events.
 */

// @ts-ignore — CDN import for standalone worker context
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

const MSG_TAG = 'vecdb' as const;

let extractor: any = null;
let currentModel: string | null = null;
let currentDtype: string | null = null;
let currentDim: number | null = null;

function send(msg: Record<string, any>) {
    self.postMessage({ ...msg, source: MSG_TAG });
}

function sendTransfer(msg: Record<string, any>, transfer: Transferable[]) {
    self.postMessage({ ...msg, source: MSG_TAG }, transfer);
}

self.addEventListener('message', async (event: MessageEvent) => {
    const { type, id } = event.data;

    try {
        switch (type) {
            case 'load': {
                const { modelId, dtype } = event.data;

                // Model diffing — skip if already loaded
                if (extractor && currentModel === modelId && currentDtype === dtype && currentDim) {
                    send({ status: 'ready', dim: currentDim, modelId, dtype, cached: true });
                    break;
                }

                if (extractor) {
                    try { await extractor.dispose(); } catch {}
                    extractor = null;
                }

                currentModel = modelId;
                currentDtype = dtype;
                currentDim = null;

                extractor = await pipeline('feature-extraction', modelId, {
                    dtype,
                    progress_callback: (p: any) => {
                        if (p.status === 'progress' && p.total > 0) {
                            send({ status: 'dl-progress', file: p.file, loaded: p.loaded, total: p.total });
                        }
                    },
                });

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
                    [float32.buffer],
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
                send({ status: 'unloaded' });
                break;
            }

            case 'get-model': {
                send({ status: 'model-info', modelId: currentModel, dtype: currentDtype, dim: currentDim, loaded: !!extractor });
                break;
            }
        }
    } catch (err: any) {
        send({ status: 'error', id, message: err.message || String(err) });
    }
});
