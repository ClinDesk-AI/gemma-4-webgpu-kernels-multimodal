import { runMultimodalEmbedderCpu, runMultimodalEmbedderWebGpu } from "../embedder/index.js";
import { navigatorGpu } from "../gpu/runtime.js";
import { fetchSafetensorsHeader, float32FromBytes, loadSafetensorsTensorData } from "../io/safetensors.js";
import { GEMMA4_MEDIA_MODEL_ID } from "../model.js";
import { mediaArtifactUrls } from "../probes/media.js";
import { Gemma4MediaEmbedderKernelProbe } from "../types.js";
import { checksumBytes, checksumFloats, deterministicMultimodalHiddenStates, maxAbsDifference, roundedSample } from "../utils/math.js";

export async function probeGemma4MediaEmbedderKernel(
  kind: "vision" | "audio",
  signal?: AbortSignal,
): Promise<Gemma4MediaEmbedderKernelProbe> {
  const started = performance.now();
  const projectionName = kind === "vision"
    ? "model.embed_vision.embedding_projection.weight"
    : "model.embed_audio.embedding_projection.weight";
  const rows = 4;
  const inputDim = kind === "vision" ? 768 : 1536;
  const outputDim = 1536;
  const epsilon = 0.000001;
  try {
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
      const header = await fetchSafetensorsHeader(urls.safetensors, signal);
      const projectionData = await loadSafetensorsTensorData(urls.safetensors, header, projectionName, signal);
      if (projectionData.dtype !== "F32" ||
        projectionData.shape.length !== 2 ||
        projectionData.shape[0] !== outputDim ||
        projectionData.shape[1] !== inputDim
      ) {
        throw new Error(`Unexpected ${kind} embedder projection tensor: ${projectionData.dtype} [${projectionData.shape.join(", ")}].`);
      }

      const weights = float32FromBytes(projectionData.bytes);
      const hiddenStates = deterministicMultimodalHiddenStates(rows, inputDim);
      const cpuOutput = runMultimodalEmbedderCpu(hiddenStates, weights, rows, inputDim, outputDim, epsilon);
      const gpuOutput = await runMultimodalEmbedderWebGpu(device, {
        hiddenStates,
        weights,
        rows,
        inputDim,
        outputDim,
        epsilon,
        label: `gemma4-${kind}-embedder`,
      });
      const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);

      return {
        ok: maxAbsDiff <= 0.001,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        kind,
        durationMs: Math.round(performance.now() - started),
        tensor: {
          name: projectionData.name,
          dtype: projectionData.dtype,
          shape: projectionData.shape,
          dataBytes: projectionData.dataBytes,
          fromCache: projectionData.fromCache,
          checksum: checksumBytes(projectionData.bytes),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          inputDim,
          outputDim,
          epsilon,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff,
          cpuChecksum: checksumFloats(cpuOutput),
          gpuChecksum: checksumFloats(gpuOutput),
          firstCpuValues: roundedSample(cpuOutput),
          firstGpuValues: roundedSample(gpuOutput),
        },
      };
    } finally {
      device.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      kind,
      durationMs: Math.round(performance.now() - started),
      tensor: null,
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        inputDim: 0,
        outputDim: 0,
        epsilon,
        workgroupSize: 64,
      },
      comparison: {
        maxAbsDiff: null,
        cpuChecksum: null,
        gpuChecksum: null,
        firstCpuValues: [],
        firstGpuValues: [],
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
