import { computeMultimodalEmbedderWebGpuOnly } from "../embedder/index.js";
import { navigatorGpu } from "../gpu/runtime.js";
import { fetchSafetensorsHeader, float32FromBf16Bytes, loadSafetensorsTensorData } from "../io/safetensors.js";
import { GEMMA4_MEDIA_MODEL_ID, VISION_ENCODER_LAYER_COUNT, checkedVisionLayerCount } from "../model.js";
import { preprocessGemma4ImageBlob, validImagePatchInput } from "../preprocess/image.js";
import { mediaArtifactUrls } from "../probes/media.js";
import { Gemma4VisionImageEmbeddingResult } from "../types.js";
import { allFinite, checksumFloats, roundedSample } from "../utils/math.js";
import { computeVisionEncoderStackWebGpuOnly, loadVisionPatchPositionRows } from "./encoder.js";
import { runVisionPatchEmbeddingWebGpu, runVisionPoolerWebGpu } from "./kernels.js";

export async function computeGemma4VisionImageEmbeddings(input: {
  blob: Blob;
  maxSoftTokens?: number;
  layerCount?: number;
  signal?: AbortSignal;
}): Promise<Gemma4VisionImageEmbeddingResult> {
  const started = performance.now();
  const hiddenSize = 768;
  const intermediateSize = 3072;
  const heads = 12;
  const headDim = 64;
  const textHiddenSize = 1536;
  const poolingKernelSize = 3;
  const epsilon = 0.000001;
  const layerCount = input.layerCount ?? VISION_ENCODER_LAYER_COUNT;
  try {
    const checkedLayerCount = checkedVisionLayerCount(layerCount);
    const gpu = navigatorGpu();
    if (!gpu) {
      throw new Error("WebGPU is not available in this browser.");
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU adapter request returned null.");
    }
    const device = await adapter.requestDevice();
    try {
      const urls = mediaArtifactUrls();
      const header = await fetchSafetensorsHeader(urls.safetensors, input.signal);
      const preprocessed = await preprocessGemma4ImageBlob(input.blob, {
        maxSoftTokens: input.maxSoftTokens,
        poolingKernelSize,
      });
      const patchInput = validImagePatchInput(preprocessed);
      const patchProjectionName = "model.vision_tower.patch_embedder.input_proj.weight";
      const positionTableName = "model.vision_tower.patch_embedder.position_embedding_table";
      const projection = await loadSafetensorsTensorData(
        urls.safetensors,
        header,
        patchProjectionName,
        input.signal,
      );
      if (projection.dtype !== "BF16" ||
        projection.shape.length !== 2 ||
        projection.shape[0] !== hiddenSize ||
        projection.shape[1] !== preprocessed.patchPixels
      ) {
        throw new Error(`Unexpected image embedding patch projection tensor: ${projection.dtype} [${projection.shape.join(", ")}].`);
      }
      const positionRows = await loadVisionPatchPositionRows({
        urls,
        header,
        tensorName: positionTableName,
        positions: patchInput.positions,
        hiddenSize,
        signal: input.signal,
      });
      const patchEmbeddings = await runVisionPatchEmbeddingWebGpu(device, {
        patchValues: patchInput.patchValues,
        projectionWeights: float32FromBf16Bytes(projection.bytes),
        positions: patchInput.positions,
        positionXRows: positionRows.xRows,
        positionYRows: positionRows.yRows,
        patchPixels: preprocessed.patchPixels,
        hiddenSize,
        outputRows: hiddenSize,
      });
      const encodedPatches = await computeVisionEncoderStackWebGpuOnly({
        device,
        urls,
        header,
        layerCount: checkedLayerCount,
        hiddenStates: patchEmbeddings,
        positions: patchInput.positions,
        rows: patchInput.positions.length,
        hiddenSize,
        intermediateSize,
        heads,
        headDim,
        epsilon,
        signal: input.signal,
      });
      const pooled = await runVisionPoolerWebGpu(device, {
        hiddenStates: encodedPatches,
        positions: patchInput.positions,
        outputLength: preprocessed.numSoftTokens,
        hiddenSize,
        poolingKernelSize,
      });
      const embeddings = await computeMultimodalEmbedderWebGpuOnly({
        device,
        urls,
        header,
        kind: "vision",
        hiddenStates: pooled,
        rows: preprocessed.numSoftTokens,
        inputDim: hiddenSize,
        outputDim: textHiddenSize,
        epsilon,
        signal: input.signal,
      });
      const finite = allFinite(embeddings);
      return {
        ok: finite,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        source: preprocessed.layout.input,
        layout: preprocessed.layout,
        layerCount: checkedLayerCount,
        rows: preprocessed.numSoftTokens,
        dim: textHiddenSize,
        embeddings,
        finite,
        checksum: checksumFloats(embeddings),
        firstValues: roundedSample(embeddings),
      };
    } finally {
      device.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      source: { width: 0, height: 0 },
      layout: null,
      layerCount,
      rows: 0,
      dim: 0,
      embeddings: new Float32Array(),
      finite: false,
      checksum: null,
      firstValues: [],
      error: error instanceof Error ? error.message : String(error),
    };
  }
}