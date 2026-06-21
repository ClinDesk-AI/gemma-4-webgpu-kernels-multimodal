import { computeMultimodalEmbedder } from "../embedder/index.js";
import { navigatorGpu } from "../gpu/runtime.js";
import { assertBf16Vector, fetchSafetensorsHeader, float32FromBf16Bytes, float32FromBytes, int8FromBytes, loadSafetensorsTensorData, scalarF32FromTensor } from "../io/safetensors.js";
import { runQatI8LinearCpu, runQatI8LinearWebGpu } from "../kernels/qat.js";
import { GEMMA4_MEDIA_MODEL_ID, VISION_ATTENTION_NORM_TENSORS, VISION_ATTENTION_OUTPUT_PROJECTION_TENSORS, VISION_ENCODER_LAYER0_NORM_TENSORS, VISION_ENCODER_LAYER_COUNT, checkedVisionLayerCount, checkedVisionLayerIndex } from "../model.js";
import { createDeterministicImageCanvas, preprocessGemma4ImageSource, validImagePatchInput } from "../preprocess/image.js";
import { mediaArtifactUrls } from "../probes/media.js";
import { probeGemma4MediaEmbedderKernel } from "../embedder/probes.js";
import { Gemma4MediaKernelProbe, Gemma4VisionAttentionBodyKernelProbe, Gemma4VisionAttentionNormKernelProbe, Gemma4VisionAttentionNormResult, Gemma4VisionAttentionOutputProjectionKernelProbe, Gemma4VisionAttentionProjectionKernelProbe, Gemma4VisionAttentionProjectionName, Gemma4VisionAttentionProjectionResult, Gemma4VisionAttentionRopeKernelProbe, Gemma4VisionAttentionRopeResult, Gemma4VisionEmbedderKernelProbe, Gemma4VisionEncoderLayerKernelProbe, Gemma4VisionEncoderStackKernelProbe, Gemma4VisionFeedForwardKernelProbe, Gemma4VisionImageFeaturesKernelProbe, Gemma4VisionImageGpuPathKernelProbe, Gemma4VisionPatchEmbeddingKernelProbe, Gemma4VisionPostAttentionKernelProbe, Gemma4VisionRmsNormKernelProbe, SafetensorsHeader } from "../types.js";
import { allFinite, checksumBytes, checksumFloats, deterministicMultimodalHiddenStates, deterministicPatchValues, deterministicVisionHiddenStates, maxAbsDifference, roundedSample, tensorSummaryForProbe, visionRopePositions } from "../utils/math.js";
import { computeVisionAttentionNorm, computeVisionAttentionOutputProjection, computeVisionAttentionProjection, computeVisionEncoderLayer, computeVisionEncoderStack, computeVisionMlp, loadVisionPatchPositionRows, visionMlpProjectionSummary } from "./encoder.js";
import { runResidualAddCpu, runResidualAddWebGpu, runVisionAttentionBodyCpu, runVisionAttentionBodyWebGpu, runVisionPatchEmbeddingCpu, runVisionPatchEmbeddingWebGpu, runVisionPoolerCpu, runVisionPoolerWebGpu, runVisionRmsNormCpu, runVisionRmsNormWebGpu, runVisionRopeCpu, runVisionRopeWebGpu } from "./kernels.js";

export async function probeGemma4VisionPatchEmbeddingKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionPatchEmbeddingKernelProbe> {
  const started = performance.now();
  const inputProjectionName = "model.vision_tower.patch_embedder.input_proj.weight";
  const positionTableName = "model.vision_tower.patch_embedder.position_embedding_table";
  const positions: Array<{ x: number; y: number }> = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 0, y: 1 },
    { x: 1, y: 1 },
  ];
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
      const projection = await loadSafetensorsTensorData(urls.safetensors, header, inputProjectionName, signal);
      if (projection.dtype !== "BF16") {
        throw new Error(`Expected BF16 patch projection tensor, got ${projection.dtype}.`);
      }
      if (projection.shape.length !== 2 || projection.shape[0] !== projection.shape[1]) {
        throw new Error(`Expected square 2-D patch projection tensor, got [${projection.shape.join(", ")}].`);
      }

      const [hiddenSize, patchPixels] = projection.shape;
      const outputRows = Math.min(64, hiddenSize);
      const projectionWeights = float32FromBf16Bytes(projection.bytes);
      const positionRows = await loadVisionPatchPositionRows({
        urls,
        header,
        tensorName: positionTableName,
        positions,
        hiddenSize,
        signal,
      });

      const patchValues = deterministicPatchValues(positions.length, patchPixels);
      const cpuOutput = runVisionPatchEmbeddingCpu({
        patchValues,
        projectionWeights,
        positions,
        positionXRows: positionRows.xRows,
        positionYRows: positionRows.yRows,
        patchPixels,
        hiddenSize,
        outputRows,
      });
      const gpuOutput = await runVisionPatchEmbeddingWebGpu(device, {
        patchValues,
        projectionWeights,
        positions,
        positionXRows: positionRows.xRows,
        positionYRows: positionRows.yRows,
        patchPixels,
        hiddenSize,
        outputRows,
      });
      const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);

      return {
        ok: maxAbsDiff <= 0.001,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        tensors: {
          inputProjection: {
            name: projection.name,
            dtype: projection.dtype,
            shape: projection.shape,
            dataBytes: projection.dataBytes,
            fromCache: projection.fromCache,
            checksum: checksumBytes(projection.bytes),
          },
          positionRows: positionRows.rowSlices.map((row) => ({
            axis: row.axis,
            indexStart: row.indexStart,
            rowCount: row.rowCount,
            dataBytes: row.bytes.dataBytes,
            fromCache: row.bytes.fromCache,
            checksum: checksumBytes(row.bytes.bytes),
          })),
        },
        gpu: {
          adapter: true,
          device: true,
          patches: positions.length,
          patchPixels,
          hiddenSize,
          outputRows,
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
      durationMs: Math.round(performance.now() - started),
      tensors: {
        inputProjection: null,
        positionRows: [],
      },
      gpu: {
        adapter: false,
        device: false,
        patches: 0,
        patchPixels: 0,
        hiddenSize: 0,
        outputRows: 0,
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

export async function probeGemma4VisionRmsNormKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionRmsNormKernelProbe> {
  const started = performance.now();
  const weightName = VISION_ENCODER_LAYER0_NORM_TENSORS.inputLayerNorm;
  const rows = 4;
  const hiddenSize = 768;
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
      const weightData = await loadSafetensorsTensorData(urls.safetensors, header, weightName, signal);
      if (weightData.dtype !== "BF16" || weightData.shape.length !== 1 || weightData.shape[0] !== hiddenSize) {
        throw new Error(`Unexpected RMSNorm weight tensor: ${weightData.dtype} [${weightData.shape.join(", ")}].`);
      }

      const weights = float32FromBf16Bytes(weightData.bytes);
      const hiddenStates = deterministicVisionHiddenStates(rows, hiddenSize);
      const cpuOutput = runVisionRmsNormCpu(hiddenStates, weights, rows, hiddenSize, epsilon);
      const gpuOutput = await runVisionRmsNormWebGpu(device, hiddenStates, weights, rows, hiddenSize, epsilon);
      const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);

      return {
        ok: maxAbsDiff <= 0.001,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        tensor: {
          name: weightData.name,
          dtype: weightData.dtype,
          shape: weightData.shape,
          dataBytes: weightData.dataBytes,
          fromCache: weightData.fromCache,
          checksum: checksumBytes(weightData.bytes),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          epsilon,
          workgroupSize: 256,
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
      durationMs: Math.round(performance.now() - started),
      tensor: null,
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        epsilon,
        workgroupSize: 256,
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

export async function probeGemma4VisionAttentionProjectionKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionAttentionProjectionKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const inputDim = 768;
  const outputDim = 768;
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
      const hiddenStates = deterministicMultimodalHiddenStates(rows, inputDim);
      const projections: Gemma4VisionAttentionProjectionResult[] = [];
      for (const projection of ["q_proj", "k_proj", "v_proj"] as const) {
        projections.push(await probeVisionAttentionProjection({
          device,
          urls,
          header,
          projection,
          hiddenStates,
          rows,
          inputDim,
          outputDim,
          signal,
        }));
      }

      return {
        ok: projections.every((result) => result.comparison.maxAbsDiff !== null && result.comparison.maxAbsDiff <= 0.001),
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        projections,
        gpu: {
          adapter: true,
          device: true,
          rows,
          inputDim,
          outputDim,
          workgroupSize: 64,
        },
      };
    } finally {
      device.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      projections: [],
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        inputDim: 0,
        outputDim: 0,
        workgroupSize: 64,
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function probeGemma4VisionAttentionNormKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionAttentionNormKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const inputDim = 768;
  const heads = 12;
  const headDim = 64;
  const outputDim = heads * headDim;
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
      const hiddenStates = deterministicMultimodalHiddenStates(rows, inputDim);
      const norms: Gemma4VisionAttentionNormResult[] = [];
      for (const projection of ["q_proj", "k_proj", "v_proj"] as const) {
        norms.push(await probeVisionAttentionNorm({
          device,
          urls,
          header,
          projection,
          hiddenStates,
          rows,
          inputDim,
          outputDim,
          heads,
          headDim,
          epsilon,
          signal,
        }));
      }

      return {
        ok: norms.every((result) => result.comparison.maxAbsDiff !== null && result.comparison.maxAbsDiff <= 0.001),
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        norms,
        gpu: {
          adapter: true,
          device: true,
          rows,
          heads,
          headDim,
          epsilon,
          workgroupSize: 1,
        },
      };
    } finally {
      device.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      norms: [],
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        heads: 0,
        headDim: 0,
        epsilon,
        workgroupSize: 1,
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function probeGemma4VisionAttentionRopeKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionAttentionRopeKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const inputDim = 768;
  const heads = 12;
  const headDim = 64;
  const outputDim = heads * headDim;
  const epsilon = 0.000001;
  const ropeTheta = 100;
  const positions = visionRopePositions(rows);
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
      const hiddenStates = deterministicMultimodalHiddenStates(rows, inputDim);
      const ropes: Gemma4VisionAttentionRopeResult[] = [];
      for (const projection of ["q_proj", "k_proj"] as const) {
        ropes.push(await probeVisionAttentionRope({
          device,
          urls,
          header,
          projection,
          hiddenStates,
          rows,
          inputDim,
          outputDim,
          heads,
          headDim,
          epsilon,
          ropeTheta,
          positions,
          signal,
        }));
      }

      return {
        ok: ropes.every((result) => result.comparison.maxAbsDiff !== null && result.comparison.maxAbsDiff <= 0.001),
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        ropes,
        positions,
        gpu: {
          adapter: true,
          device: true,
          rows,
          heads,
          headDim,
          ropeTheta,
          workgroupSize: 64,
        },
      };
    } finally {
      device.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      ropes: [],
      positions,
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        heads: 0,
        headDim: 0,
        ropeTheta,
        workgroupSize: 64,
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function probeGemma4VisionAttentionBodyKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionAttentionBodyKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const inputDim = 768;
  const heads = 12;
  const headDim = 64;
  const outputDim = heads * headDim;
  const epsilon = 0.000001;
  const ropeTheta = 100;
  const scaling = 1;
  const positions = visionRopePositions(rows);
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
      const hiddenStates = deterministicMultimodalHiddenStates(rows, inputDim);
      const common = {
        device,
        urls,
        header,
        hiddenStates,
        rows,
        inputDim,
        outputDim,
        heads,
        headDim,
        epsilon,
        signal,
      };
      const [queryNorm, keyNorm, valueNorm] = await Promise.all([
        computeVisionAttentionNorm({ ...common, projection: "q_proj" }),
        computeVisionAttentionNorm({ ...common, projection: "k_proj" }),
        computeVisionAttentionNorm({ ...common, projection: "v_proj" }),
      ]);
      const queryCpu = runVisionRopeCpu(queryNorm.cpuOutput, positions, rows, heads, headDim, ropeTheta);
      const keyCpu = runVisionRopeCpu(keyNorm.cpuOutput, positions, rows, heads, headDim, ropeTheta);
      const queryGpu = await runVisionRopeWebGpu(device, queryNorm.gpuOutput, positions, rows, heads, headDim, ropeTheta);
      const keyGpu = await runVisionRopeWebGpu(device, keyNorm.gpuOutput, positions, rows, heads, headDim, ropeTheta);
      const cpuOutput = runVisionAttentionBodyCpu({
        query: queryCpu,
        key: keyCpu,
        value: valueNorm.cpuOutput,
        rows,
        heads,
        headDim,
        scaling,
      });
      const gpuOutput = await runVisionAttentionBodyWebGpu(device, {
        query: queryGpu,
        key: keyGpu,
        value: valueNorm.gpuOutput,
        rows,
        heads,
        headDim,
        scaling,
      });
      const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);

      return {
        ok: maxAbsDiff <= 0.001,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        gpu: {
          adapter: true,
          device: true,
          rows,
          heads,
          headDim,
          scaling,
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
      durationMs: Math.round(performance.now() - started),
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        heads: 0,
        headDim: 0,
        scaling,
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

export async function probeGemma4VisionAttentionOutputProjectionKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionAttentionOutputProjectionKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const inputDim = 768;
  const heads = 12;
  const headDim = 64;
  const outputDim = heads * headDim;
  const epsilon = 0.000001;
  const ropeTheta = 100;
  const scaling = 1;
  const positions = visionRopePositions(rows);
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
      const hiddenStates = deterministicMultimodalHiddenStates(rows, inputDim);
      const common = {
        device,
        urls,
        header,
        hiddenStates,
        rows,
        inputDim,
        outputDim,
        heads,
        headDim,
        epsilon,
        signal,
      };
      const [queryNorm, keyNorm, valueNorm] = await Promise.all([
        computeVisionAttentionNorm({ ...common, projection: "q_proj" }),
        computeVisionAttentionNorm({ ...common, projection: "k_proj" }),
        computeVisionAttentionNorm({ ...common, projection: "v_proj" }),
      ]);
      const queryCpu = runVisionRopeCpu(queryNorm.cpuOutput, positions, rows, heads, headDim, ropeTheta);
      const keyCpu = runVisionRopeCpu(keyNorm.cpuOutput, positions, rows, heads, headDim, ropeTheta);
      const queryGpu = await runVisionRopeWebGpu(device, queryNorm.gpuOutput, positions, rows, heads, headDim, ropeTheta);
      const keyGpu = await runVisionRopeWebGpu(device, keyNorm.gpuOutput, positions, rows, heads, headDim, ropeTheta);
      const attentionCpu = runVisionAttentionBodyCpu({
        query: queryCpu,
        key: keyCpu,
        value: valueNorm.cpuOutput,
        rows,
        heads,
        headDim,
        scaling,
      });
      const attentionGpu = await runVisionAttentionBodyWebGpu(device, {
        query: queryGpu,
        key: keyGpu,
        value: valueNorm.gpuOutput,
        rows,
        heads,
        headDim,
        scaling,
      });
      const names = VISION_ATTENTION_OUTPUT_PROJECTION_TENSORS;
      const [weightData, weightScaleData, inputScaleData, outputScaleData] = await Promise.all([
        loadSafetensorsTensorData(urls.safetensors, header, names.weight, signal),
        loadSafetensorsTensorData(urls.safetensors, header, names.weightScale, signal),
        loadSafetensorsTensorData(urls.safetensors, header, names.inputScale, signal),
        loadSafetensorsTensorData(urls.safetensors, header, names.outputScale, signal),
      ]);
      if (weightData.dtype !== "I8" ||
        weightData.shape.length !== 2 ||
        weightData.shape[0] !== outputDim ||
        weightData.shape[1] !== outputDim
      ) {
        throw new Error(`Unexpected attention o_proj weight tensor: ${weightData.dtype} [${weightData.shape.join(", ")}].`);
      }
      if (weightScaleData.dtype !== "F32" ||
        weightScaleData.shape.length !== 2 ||
        weightScaleData.shape[0] !== outputDim ||
        weightScaleData.shape[1] !== 1
      ) {
        throw new Error(`Unexpected attention o_proj weight scale tensor: ${weightScaleData.dtype} [${weightScaleData.shape.join(", ")}].`);
      }
      const inputActivationScale = scalarF32FromTensor(inputScaleData, names.inputScale);
      const outputActivationScale = scalarF32FromTensor(outputScaleData, names.outputScale);
      const weightScales = float32FromBytes(weightScaleData.bytes);
      const cpuOutput = runQatI8LinearCpu({
        input: attentionCpu,
        weights: int8FromBytes(weightData.bytes),
        weightScales,
        rows,
        inputDim: outputDim,
        outputDim,
        inputActivationScale,
        outputActivationScale,
      });
      const gpuOutput = await runQatI8LinearWebGpu(device, {
        input: attentionGpu,
        weights: weightData.bytes,
        weightScales,
        rows,
        inputDim: outputDim,
        outputDim,
        inputActivationScale,
        outputActivationScale,
        label: "gemma4-vision-attention-o-proj",
      });
      const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);

      return {
        ok: maxAbsDiff <= 0.001,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        tensors: {
          weight: {
            name: weightData.name,
            dtype: weightData.dtype,
            shape: weightData.shape,
            dataBytes: weightData.dataBytes,
            fromCache: weightData.fromCache,
            checksum: checksumBytes(weightData.bytes),
          },
          weightScale: {
            name: weightScaleData.name,
            dtype: weightScaleData.dtype,
            shape: weightScaleData.shape,
            dataBytes: weightScaleData.dataBytes,
            fromCache: weightScaleData.fromCache,
            checksum: checksumBytes(weightScaleData.bytes),
          },
        },
        quantization: {
          inputActivationScale,
          outputActivationScale,
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          inputDim: outputDim,
          outputDim,
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
      durationMs: Math.round(performance.now() - started),
      tensors: {
        weight: null,
        weightScale: null,
      },
      quantization: {
        inputActivationScale: null,
        outputActivationScale: null,
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        inputDim: 0,
        outputDim: 0,
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

export async function probeGemma4VisionPostAttentionKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionPostAttentionKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const hiddenSize = 768;
  const heads = 12;
  const headDim = 64;
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
      const [inputNormData, postAttentionNormData] = await Promise.all([
        loadSafetensorsTensorData(
          urls.safetensors,
          header,
          VISION_ENCODER_LAYER0_NORM_TENSORS.inputLayerNorm,
          signal,
        ),
        loadSafetensorsTensorData(
          urls.safetensors,
          header,
          VISION_ENCODER_LAYER0_NORM_TENSORS.postAttentionLayerNorm,
          signal,
        ),
      ]);
      assertBf16Vector(inputNormData, hiddenSize, "vision input layernorm");
      assertBf16Vector(postAttentionNormData, hiddenSize, "vision post-attention layernorm");

      const residual = deterministicMultimodalHiddenStates(rows, hiddenSize);
      const inputNormWeights = float32FromBf16Bytes(inputNormData.bytes);
      const postAttentionNormWeights = float32FromBf16Bytes(postAttentionNormData.bytes);
      const attentionInputCpu = runVisionRmsNormCpu(residual, inputNormWeights, rows, hiddenSize, epsilon);
      const attentionInputGpu = await runVisionRmsNormWebGpu(device, residual, inputNormWeights, rows, hiddenSize, epsilon);
      const attentionOutput = await computeVisionAttentionOutputProjection({
        device,
        urls,
        header,
        hiddenStates: attentionInputCpu,
        gpuHiddenStates: attentionInputGpu,
        rows,
        inputDim: hiddenSize,
        outputDim: hiddenSize,
        heads,
        headDim,
        epsilon,
        signal,
      });
      const postAttentionCpu = runVisionRmsNormCpu(
        attentionOutput.cpuOutput,
        postAttentionNormWeights,
        rows,
        hiddenSize,
        epsilon,
      );
      const postAttentionGpu = await runVisionRmsNormWebGpu(
        device,
        attentionOutput.gpuOutput,
        postAttentionNormWeights,
        rows,
        hiddenSize,
        epsilon,
      );
      const cpuOutput = runResidualAddCpu(residual, postAttentionCpu);
      const gpuOutput = await runResidualAddWebGpu(device, residual, postAttentionGpu, "gemma4-vision-post-attention-residual");
      const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);

      return {
        ok: maxAbsDiff <= 0.001,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        tensors: {
          inputLayerNorm: {
            name: inputNormData.name,
            dtype: inputNormData.dtype,
            shape: inputNormData.shape,
            dataBytes: inputNormData.dataBytes,
            fromCache: inputNormData.fromCache,
            checksum: checksumBytes(inputNormData.bytes),
          },
          postAttentionLayerNorm: {
            name: postAttentionNormData.name,
            dtype: postAttentionNormData.dtype,
            shape: postAttentionNormData.shape,
            dataBytes: postAttentionNormData.dataBytes,
            fromCache: postAttentionNormData.fromCache,
            checksum: checksumBytes(postAttentionNormData.bytes),
          },
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          heads,
          headDim,
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
      durationMs: Math.round(performance.now() - started),
      tensors: {
        inputLayerNorm: null,
        postAttentionLayerNorm: null,
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        heads: 0,
        headDim: 0,
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

export async function probeGemma4VisionFeedForwardKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionFeedForwardKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const hiddenSize = 768;
  const intermediateSize = 3072;
  const heads = 12;
  const headDim = 64;
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
      const [inputNormData, postAttentionNormData, preFeedForwardNormData, postFeedForwardNormData] = await Promise.all([
        loadSafetensorsTensorData(
          urls.safetensors,
          header,
          VISION_ENCODER_LAYER0_NORM_TENSORS.inputLayerNorm,
          signal,
        ),
        loadSafetensorsTensorData(
          urls.safetensors,
          header,
          VISION_ENCODER_LAYER0_NORM_TENSORS.postAttentionLayerNorm,
          signal,
        ),
        loadSafetensorsTensorData(
          urls.safetensors,
          header,
          VISION_ENCODER_LAYER0_NORM_TENSORS.preFeedForwardLayerNorm,
          signal,
        ),
        loadSafetensorsTensorData(
          urls.safetensors,
          header,
          VISION_ENCODER_LAYER0_NORM_TENSORS.postFeedForwardLayerNorm,
          signal,
        ),
      ]);
      assertBf16Vector(inputNormData, hiddenSize, "vision input layernorm");
      assertBf16Vector(postAttentionNormData, hiddenSize, "vision post-attention layernorm");
      assertBf16Vector(preFeedForwardNormData, hiddenSize, "vision pre-feedforward layernorm");
      assertBf16Vector(postFeedForwardNormData, hiddenSize, "vision post-feedforward layernorm");

      const residual = deterministicMultimodalHiddenStates(rows, hiddenSize);
      const inputNormWeights = float32FromBf16Bytes(inputNormData.bytes);
      const postAttentionNormWeights = float32FromBf16Bytes(postAttentionNormData.bytes);
      const preFeedForwardNormWeights = float32FromBf16Bytes(preFeedForwardNormData.bytes);
      const postFeedForwardNormWeights = float32FromBf16Bytes(postFeedForwardNormData.bytes);
      const attentionInputCpu = runVisionRmsNormCpu(residual, inputNormWeights, rows, hiddenSize, epsilon);
      const attentionInputGpu = await runVisionRmsNormWebGpu(device, residual, inputNormWeights, rows, hiddenSize, epsilon);
      const attentionOutput = await computeVisionAttentionOutputProjection({
        device,
        urls,
        header,
        hiddenStates: attentionInputCpu,
        gpuHiddenStates: attentionInputGpu,
        rows,
        inputDim: hiddenSize,
        outputDim: hiddenSize,
        heads,
        headDim,
        epsilon,
        signal,
      });
      const postAttentionCpu = runVisionRmsNormCpu(
        attentionOutput.cpuOutput,
        postAttentionNormWeights,
        rows,
        hiddenSize,
        epsilon,
      );
      const postAttentionGpu = await runVisionRmsNormWebGpu(
        device,
        attentionOutput.gpuOutput,
        postAttentionNormWeights,
        rows,
        hiddenSize,
        epsilon,
      );
      const feedForwardResidualCpu = runResidualAddCpu(residual, postAttentionCpu);
      const feedForwardResidualGpu = await runResidualAddWebGpu(
        device,
        residual,
        postAttentionGpu,
        "gemma4-vision-feedforward-residual",
      );
      const feedForwardInputCpu = runVisionRmsNormCpu(
        feedForwardResidualCpu,
        preFeedForwardNormWeights,
        rows,
        hiddenSize,
        epsilon,
      );
      const feedForwardInputGpu = await runVisionRmsNormWebGpu(
        device,
        feedForwardResidualGpu,
        preFeedForwardNormWeights,
        rows,
        hiddenSize,
        epsilon,
      );
      const mlpOutput = await computeVisionMlp({
        device,
        urls,
        header,
        hiddenStates: feedForwardInputCpu,
        gpuHiddenStates: feedForwardInputGpu,
        rows,
        hiddenSize,
        intermediateSize,
        signal,
      });
      const postFeedForwardCpu = runVisionRmsNormCpu(
        mlpOutput.cpuOutput,
        postFeedForwardNormWeights,
        rows,
        hiddenSize,
        epsilon,
      );
      const postFeedForwardGpu = await runVisionRmsNormWebGpu(
        device,
        mlpOutput.gpuOutput,
        postFeedForwardNormWeights,
        rows,
        hiddenSize,
        epsilon,
      );
      const cpuOutput = runResidualAddCpu(feedForwardResidualCpu, postFeedForwardCpu);
      const gpuOutput = await runResidualAddWebGpu(
        device,
        feedForwardResidualGpu,
        postFeedForwardGpu,
        "gemma4-vision-feedforward-output-residual",
      );
      const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);

      return {
        ok: maxAbsDiff <= 0.001,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        tensors: {
          preFeedForwardLayerNorm: {
            name: preFeedForwardNormData.name,
            dtype: preFeedForwardNormData.dtype,
            shape: preFeedForwardNormData.shape,
            dataBytes: preFeedForwardNormData.dataBytes,
            fromCache: preFeedForwardNormData.fromCache,
            checksum: checksumBytes(preFeedForwardNormData.bytes),
          },
          postFeedForwardLayerNorm: {
            name: postFeedForwardNormData.name,
            dtype: postFeedForwardNormData.dtype,
            shape: postFeedForwardNormData.shape,
            dataBytes: postFeedForwardNormData.dataBytes,
            fromCache: postFeedForwardNormData.fromCache,
            checksum: checksumBytes(postFeedForwardNormData.bytes),
          },
          mlpProjections: mlpOutput.projections.map(visionMlpProjectionSummary),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          intermediateSize,
          activation: "gelu_pytorch_tanh",
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
      durationMs: Math.round(performance.now() - started),
      tensors: {
        preFeedForwardLayerNorm: null,
        postFeedForwardLayerNorm: null,
        mlpProjections: [],
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        intermediateSize: 0,
        activation: "gelu_pytorch_tanh",
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

export async function probeGemma4VisionEncoderLayerKernel(
  layerIndex = 1,
  signal?: AbortSignal,
): Promise<Gemma4VisionEncoderLayerKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const hiddenSize = 768;
  const intermediateSize = 3072;
  const heads = 12;
  const headDim = 64;
  const epsilon = 0.000001;
  try {
    const checkedLayerIndex = checkedVisionLayerIndex(layerIndex);
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
      const layer = await computeVisionEncoderLayer({
        device,
        urls,
        header,
        layerIndex: checkedLayerIndex,
        hiddenStates: deterministicMultimodalHiddenStates(rows, hiddenSize),
        rows,
        hiddenSize,
        intermediateSize,
        heads,
        headDim,
        epsilon,
        signal,
      });

      return {
        ok: layer.maxAbsDiff <= 0.001,
        layerIndex: checkedLayerIndex,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        tensors: {
          preFeedForwardLayerNorm: tensorSummaryForProbe(layer.preFeedForwardNormData),
          postFeedForwardLayerNorm: tensorSummaryForProbe(layer.postFeedForwardNormData),
          mlpProjections: layer.mlpOutput.projections.map(visionMlpProjectionSummary),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          intermediateSize,
          activation: "gelu_pytorch_tanh",
          epsilon,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff: layer.maxAbsDiff,
          cpuChecksum: checksumFloats(layer.cpuOutput),
          gpuChecksum: checksumFloats(layer.gpuOutput),
          firstCpuValues: roundedSample(layer.cpuOutput),
          firstGpuValues: roundedSample(layer.gpuOutput),
        },
      };
    } finally {
      device.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      layerIndex,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      tensors: {
        preFeedForwardLayerNorm: null,
        postFeedForwardLayerNorm: null,
        mlpProjections: [],
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        intermediateSize: 0,
        activation: "gelu_pytorch_tanh",
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

export async function probeGemma4VisionEncoderStackKernel(
  layerCount = VISION_ENCODER_LAYER_COUNT,
  signal?: AbortSignal,
): Promise<Gemma4VisionEncoderStackKernelProbe> {
  const started = performance.now();
  const rows = 4;
  const hiddenSize = 768;
  const intermediateSize = 3072;
  const heads = 12;
  const headDim = 64;
  const epsilon = 0.000001;
  const tolerance = 0.01;
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
      const header = await fetchSafetensorsHeader(urls.safetensors, signal);
      const stack = await computeVisionEncoderStack({
        device,
        urls,
        header,
        layerCount: checkedLayerCount,
        hiddenStates: deterministicMultimodalHiddenStates(rows, hiddenSize),
        rows,
        hiddenSize,
        intermediateSize,
        heads,
        headDim,
        epsilon,
        signal,
      });
      return {
        ok: stack.maxAbsDiff <= tolerance && stack.layers.every((layer) => layer.maxAbsDiff <= tolerance),
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        layerCount: checkedLayerCount,
        comparisonMode: "propagated",
        layers: stack.layers,
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          intermediateSize,
          heads,
          headDim,
          epsilon,
          tolerance,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff: stack.maxAbsDiff,
          cpuChecksum: checksumFloats(stack.cpuOutput),
          gpuChecksum: checksumFloats(stack.gpuOutput),
          firstCpuValues: roundedSample(stack.cpuOutput),
          firstGpuValues: roundedSample(stack.gpuOutput),
        },
      };
    } finally {
      device.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      layerCount,
      comparisonMode: "propagated",
      layers: [],
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        intermediateSize: 0,
        heads: 0,
        headDim: 0,
        epsilon,
        tolerance,
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

export async function probeGemma4VisionImageFeaturesKernel(
  layerCount = 1,
  signal?: AbortSignal,
): Promise<Gemma4VisionImageFeaturesKernelProbe> {
  const started = performance.now();
  const source = { width: 48, height: 48 };
  const hiddenSize = 768;
  const intermediateSize = 3072;
  const heads = 12;
  const headDim = 64;
  const textHiddenSize = 1536;
  const poolingKernelSize = 3;
  const epsilon = 0.000001;
  const tolerance = 0.01;
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
      const header = await fetchSafetensorsHeader(urls.safetensors, signal);
      const image = createDeterministicImageCanvas(source.width, source.height);
      const preprocessed = await preprocessGemma4ImageSource(
        image as CanvasImageSource,
        source.width,
        source.height,
        { maxSoftTokens: 1 },
      );
      const patchInput = validImagePatchInput(preprocessed);
      const patchProjectionName = "model.vision_tower.patch_embedder.input_proj.weight";
      const positionTableName = "model.vision_tower.patch_embedder.position_embedding_table";
      const projection = await loadSafetensorsTensorData(urls.safetensors, header, patchProjectionName, signal);
      if (projection.dtype !== "BF16" ||
        projection.shape.length !== 2 ||
        projection.shape[0] !== hiddenSize ||
        projection.shape[1] !== preprocessed.patchPixels
      ) {
        throw new Error(`Unexpected image feature patch projection tensor: ${projection.dtype} [${projection.shape.join(", ")}].`);
      }
      const positionRows = await loadVisionPatchPositionRows({
        urls,
        header,
        tensorName: positionTableName,
        positions: patchInput.positions,
        hiddenSize,
        signal,
      });
      const projectionWeights = float32FromBf16Bytes(projection.bytes);
      const patchCpu = runVisionPatchEmbeddingCpu({
        patchValues: patchInput.patchValues,
        projectionWeights,
        positions: patchInput.positions,
        positionXRows: positionRows.xRows,
        positionYRows: positionRows.yRows,
        patchPixels: preprocessed.patchPixels,
        hiddenSize,
        outputRows: hiddenSize,
      });
      const patchGpu = await runVisionPatchEmbeddingWebGpu(device, {
        patchValues: patchInput.patchValues,
        projectionWeights,
        positions: patchInput.positions,
        positionXRows: positionRows.xRows,
        positionYRows: positionRows.yRows,
        patchPixels: preprocessed.patchPixels,
        hiddenSize,
        outputRows: hiddenSize,
      });
      const patchEmbeddingMaxAbsDiff = maxAbsDifference(patchCpu, patchGpu);
      const stack = await computeVisionEncoderStack({
        device,
        urls,
        header,
        layerCount: checkedLayerCount,
        hiddenStates: patchCpu,
        gpuHiddenStates: patchGpu,
        comparisonMode: "gpuAnchored",
        positions: patchInput.positions,
        rows: patchInput.positions.length,
        hiddenSize,
        intermediateSize,
        heads,
        headDim,
        epsilon,
        signal,
      });
      const poolInput = {
        hiddenStates: stack.cpuOutput,
        positions: patchInput.positions,
        outputLength: preprocessed.numSoftTokens,
        hiddenSize,
        poolingKernelSize,
      };
      const poolCpu = runVisionPoolerCpu(poolInput);
      const poolGpu = await runVisionPoolerWebGpu(device, {
        ...poolInput,
        hiddenStates: stack.gpuOutput,
      });
      const poolerMaxAbsDiff = maxAbsDifference(poolCpu, poolGpu);
      const embedder = await computeMultimodalEmbedder({
        device,
        urls,
        header,
        kind: "vision",
        hiddenStates: poolCpu,
        gpuHiddenStates: poolGpu,
        rows: preprocessed.numSoftTokens,
        inputDim: hiddenSize,
        outputDim: textHiddenSize,
        epsilon,
        signal,
      });

      const comparisonOk = patchEmbeddingMaxAbsDiff <= tolerance &&
          stack.maxAbsDiff <= tolerance &&
          poolerMaxAbsDiff <= tolerance &&
        embedder.maxAbsDiff <= tolerance;
      return {
        ok: comparisonOk,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        layerCount: checkedLayerCount,
        comparisonMode: "gpuAnchored",
        gpuPath: {
          completed: true,
          finite: allFinite(embedder.gpuOutput),
          outputRows: preprocessed.numSoftTokens,
          outputDim: textHiddenSize,
          checksum: checksumFloats(embedder.gpuOutput),
          firstValues: roundedSample(embedder.gpuOutput),
        },
        source,
        layout: preprocessed.layout,
        gpu: {
          adapter: true,
          device: true,
          patches: patchInput.positions.length,
          softTokens: preprocessed.numSoftTokens,
          hiddenSize,
          textHiddenSize,
          poolingKernelSize,
          tolerance,
          workgroupSize: 64,
        },
        comparisons: {
          patchEmbeddingMaxAbsDiff,
          encoderStackMaxAbsDiff: stack.maxAbsDiff,
          poolerMaxAbsDiff,
          embedderMaxAbsDiff: embedder.maxAbsDiff,
          finalCpuChecksum: checksumFloats(embedder.cpuOutput),
          finalGpuChecksum: checksumFloats(embedder.gpuOutput),
          firstCpuValues: roundedSample(embedder.cpuOutput),
          firstGpuValues: roundedSample(embedder.gpuOutput),
        },
        layers: stack.layers,
      };
    } finally {
      device.destroy?.();
    }
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      layerCount,
      comparisonMode: "gpuAnchored",
      gpuPath: {
        completed: false,
        finite: false,
        outputRows: 0,
        outputDim: 0,
        checksum: null,
        firstValues: [],
      },
      source,
      layout: null,
      gpu: {
        adapter: false,
        device: false,
        patches: 0,
        softTokens: 0,
        hiddenSize: 0,
        textHiddenSize: 0,
        poolingKernelSize,
        tolerance,
        workgroupSize: 64,
      },
      comparisons: {
        patchEmbeddingMaxAbsDiff: null,
        encoderStackMaxAbsDiff: null,
        poolerMaxAbsDiff: null,
        embedderMaxAbsDiff: null,
        finalCpuChecksum: null,
        finalGpuChecksum: null,
        firstCpuValues: [],
        firstGpuValues: [],
      },
      layers: [],
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function probeGemma4VisionImageGpuPathKernel(
  layerCount = VISION_ENCODER_LAYER_COUNT,
  signal?: AbortSignal,
): Promise<Gemma4VisionImageGpuPathKernelProbe> {
  const result = await probeGemma4VisionImageFeaturesKernel(layerCount, signal);
  return {
    ...result,
    strictComparisonOk: result.ok,
    ok: result.gpuPath.completed && result.gpuPath.finite,
  };
}

export async function probeVisionAttentionProjection(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  projection: Gemma4VisionAttentionProjectionName;
  hiddenStates: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  signal?: AbortSignal;
}): Promise<Gemma4VisionAttentionProjectionResult> {
  try {
    const computed = await computeVisionAttentionProjection(input);
    return {
      projection: input.projection,
      tensors: {
        weight: {
          name: computed.weightData.name,
          dtype: computed.weightData.dtype,
          shape: computed.weightData.shape,
          dataBytes: computed.weightData.dataBytes,
          fromCache: computed.weightData.fromCache,
          checksum: checksumBytes(computed.weightData.bytes),
        },
        weightScale: {
          name: computed.weightScaleData.name,
          dtype: computed.weightScaleData.dtype,
          shape: computed.weightScaleData.shape,
          dataBytes: computed.weightScaleData.dataBytes,
          fromCache: computed.weightScaleData.fromCache,
          checksum: checksumBytes(computed.weightScaleData.bytes),
        },
      },
      quantization: {
        inputActivationScale: computed.inputActivationScale,
        outputActivationScale: computed.outputActivationScale,
      },
      comparison: {
        maxAbsDiff: computed.maxAbsDiff,
        cpuChecksum: checksumFloats(computed.cpuOutput),
        gpuChecksum: checksumFloats(computed.gpuOutput),
        firstCpuValues: roundedSample(computed.cpuOutput),
        firstGpuValues: roundedSample(computed.gpuOutput),
      },
    };
  } catch (error) {
    return {
      projection: input.projection,
      tensors: {
        weight: null,
        weightScale: null,
      },
      quantization: {
        inputActivationScale: null,
        outputActivationScale: null,
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

export async function probeVisionAttentionNorm(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  projection: Gemma4VisionAttentionProjectionName;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Gemma4VisionAttentionNormResult> {
  try {
    const computed = await computeVisionAttentionNorm(input);
    return {
      projection: input.projection,
      withScale: Boolean(computed.normTensor),
      tensor: computed.normTensor
        ? {
            name: computed.normTensor.name,
            dtype: computed.normTensor.dtype,
            shape: computed.normTensor.shape,
            dataBytes: computed.normTensor.dataBytes,
            fromCache: computed.normTensor.fromCache,
            checksum: checksumBytes(computed.normTensor.bytes),
          }
        : null,
      comparison: {
        maxAbsDiff: computed.maxAbsDiff,
        cpuChecksum: checksumFloats(computed.cpuOutput),
        gpuChecksum: checksumFloats(computed.gpuOutput),
        firstCpuValues: roundedSample(computed.cpuOutput),
        firstGpuValues: roundedSample(computed.gpuOutput),
      },
    };
  } catch (error) {
    return {
      projection: input.projection,
      withScale: VISION_ATTENTION_NORM_TENSORS[input.projection] !== null,
      tensor: null,
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

export async function probeVisionAttentionRope(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  projection: "q_proj" | "k_proj";
  hiddenStates: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  heads: number;
  headDim: number;
  epsilon: number;
  ropeTheta: number;
  positions: Array<{ x: number; y: number }>;
  signal?: AbortSignal;
}): Promise<Gemma4VisionAttentionRopeResult> {
  try {
    const computed = await computeVisionAttentionNorm(input);
    const cpuOutput = runVisionRopeCpu(
      computed.cpuOutput,
      input.positions,
      input.rows,
      input.heads,
      input.headDim,
      input.ropeTheta,
    );
    const gpuOutput = await runVisionRopeWebGpu(
      input.device,
      computed.gpuOutput,
      input.positions,
      input.rows,
      input.heads,
      input.headDim,
      input.ropeTheta,
    );
    const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);
    return {
      projection: input.projection,
      comparison: {
        maxAbsDiff,
        cpuChecksum: checksumFloats(cpuOutput),
        gpuChecksum: checksumFloats(gpuOutput),
        firstCpuValues: roundedSample(cpuOutput),
        firstGpuValues: roundedSample(gpuOutput),
      },
    };
  } catch (error) {
    return {
      projection: input.projection,
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

export async function probeGemma4VisionEmbedderKernel(
  signal?: AbortSignal,
): Promise<Gemma4VisionEmbedderKernelProbe> {
  return probeGemma4MediaEmbedderKernel("vision", signal);
}
