import { audioFeedForwardProjectionSummary, audioLightConvProjectionSummary, audioSelfAttentionProjectionSummary, computeAudioEncoderLayer, computeAudioEncoderStack, computeAudioFeedForward, computeAudioLightConv, computeAudioSelfAttention } from "./encoder.js";
import { computeGemma4AudioEmbeddings } from "./index.js";
import { runAudioSubsampleProjectionCpu, runAudioSubsampleProjectionWebGpu } from "./kernels.js";
import { navigatorGpu } from "../gpu/runtime.js";
import { assertF32Tensor, fetchSafetensorsHeader, float32FromBytes, loadSafetensorsTensorData } from "../io/safetensors.js";
import { AUDIO_ENCODER_LAYER_COUNT, AUDIO_FEED_FORWARD_RESIDUAL_WEIGHT, GEMMA4_MEDIA_MODEL_ID, audioEncoderLayerNormTensors, audioFeedForwardNormTensors, audioLightConvTensors, audioSelfAttentionTensors, audioSubsampleTensorNames, checkedAudioLayerCount, checkedAudioLayerIndex } from "../model.js";
import { preprocessGemma4AudioSamples } from "../preprocess/audio.js";
import { mediaArtifactUrls } from "../probes/media.js";
import { probeGemma4MediaEmbedderKernel } from "../embedder/probes.js";
import { AudioEncoderStackStepSession, Gemma4AudioEmbedderKernelProbe, Gemma4AudioEmbeddingKernelProbe, Gemma4AudioEncoderLayerKernelProbe, Gemma4AudioEncoderStackKernelProbe, Gemma4AudioEncoderStackStepKernelProbe, Gemma4AudioFeedForwardBlockName, Gemma4AudioFeedForwardKernelProbe, Gemma4AudioLightConvKernelProbe, Gemma4AudioPreprocessProbe, Gemma4AudioSelfAttentionKernelProbe, Gemma4AudioSubsampleKernelProbe } from "../types.js";
import { allFinite, checksumFloats, deterministicMultimodalHiddenStates, maxAbsDifference, roundedSample, tensorSummaryForProbe } from "../utils/math.js";

let audioEncoderStackStepSession: AudioEncoderStackStepSession | null = null;
let nextAudioEncoderStackStepSessionId = 1;

export async function probeGemma4AudioEmbeddingKernel(
  layerCount = AUDIO_ENCODER_LAYER_COUNT,
  signal?: AbortSignal,
): Promise<Gemma4AudioEmbeddingKernelProbe> {
  const samplingRate = 16_000;
  const durationSeconds = 0.08;
  const samples = new Float32Array(Math.floor(samplingRate * durationSeconds));
  for (let index = 0; index < samples.length; index += 1) {
    const t = index / samplingRate;
    samples[index] = 0.16 * Math.sin(2 * Math.PI * 440 * t) + 0.05 * Math.sin(2 * Math.PI * 660 * t);
  }
  const result = await computeGemma4AudioEmbeddings({ samples, samplingRate, layerCount, signal });
  const { embeddings: _embeddings, ...serializable } = result;
  return serializable;
}

export async function probeGemma4AudioEmbedderKernel(
  signal?: AbortSignal,
): Promise<Gemma4AudioEmbedderKernelProbe> {
  return probeGemma4MediaEmbedderKernel("audio", signal);
}

export function probeGemma4AudioPreprocessing(): Gemma4AudioPreprocessProbe {
  try {
    const samplingRate = 16_000;
    const durationSeconds = 1;
    const samples = new Float32Array(samplingRate * durationSeconds);
    for (let index = 0; index < samples.length; index += 1) {
      const t = index / samplingRate;
      samples[index] = 0.2 * Math.sin(2 * Math.PI * 440 * t);
    }
    const result = preprocessGemma4AudioSamples(samples, { samplingRate });
    return {
      ok: true,
      source: { samples: samples.length, samplingRate },
      frames: result.frames,
      validFrames: result.validFrames,
      featureSize: result.featureSize,
      inputFeatureLength: result.inputFeatures.length,
      inputFeatureMaskLength: result.inputFeaturesMask.length,
      softTokens: result.softTokens,
      firstMaskValues: Array.from(result.inputFeaturesMask.slice(0, 12)),
      featureChecksum: checksumFloats(result.inputFeatures),
    };
  } catch (error) {
    return {
      ok: false,
      source: { samples: 0, samplingRate: 0 },
      frames: 0,
      validFrames: 0,
      featureSize: 0,
      inputFeatureLength: 0,
      inputFeatureMaskLength: 0,
      softTokens: 0,
      firstMaskValues: [],
      featureChecksum: 0,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function probeGemma4AudioSubsampleKernel(
  signal?: AbortSignal,
): Promise<Gemma4AudioSubsampleKernelProbe> {
  const started = performance.now();
  const samplingRate = 16_000;
  const durationSeconds = 0.02;
  const samples = new Float32Array(Math.floor(samplingRate * durationSeconds));
  const hiddenSize = 1024;
  const tolerance = 0.001;
  const epsilon = 0.000001;
  for (let index = 0; index < samples.length; index += 1) {
    const t = index / samplingRate;
    samples[index] = 0.2 * Math.sin(2 * Math.PI * 440 * t) + 0.08 * Math.sin(2 * Math.PI * 880 * t);
  }

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
      const tensorNames = audioSubsampleTensorNames();
      const [layer0Conv, layer0Norm, layer1Conv, layer1Norm, inputProjection] = await Promise.all([
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.layer0Conv, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.layer0Norm, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.layer1Conv, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.layer1Norm, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.inputProjection, signal),
      ]);
      assertF32Tensor(layer0Conv, [128, 1, 3, 3], "audio subsample layer0 conv");
      assertF32Tensor(layer0Norm, [128], "audio subsample layer0 norm");
      assertF32Tensor(layer1Conv, [32, 128, 3, 3], "audio subsample layer1 conv");
      assertF32Tensor(layer1Norm, [32], "audio subsample layer1 norm");
      assertF32Tensor(inputProjection, [hiddenSize, hiddenSize], "audio subsample input projection");

      const preprocessed = preprocessGemma4AudioSamples(samples, { samplingRate });
      const weights = {
        layer0Conv: float32FromBytes(layer0Conv.bytes),
        layer0Norm: float32FromBytes(layer0Norm.bytes),
        layer1Conv: float32FromBytes(layer1Conv.bytes),
        layer1Norm: float32FromBytes(layer1Norm.bytes),
        inputProjection: float32FromBytes(inputProjection.bytes),
      };
      const cpu = runAudioSubsampleProjectionCpu({
        inputFeatures: preprocessed.inputFeatures,
        inputFeaturesMask: preprocessed.inputFeaturesMask,
        frames: preprocessed.frames,
        featureSize: preprocessed.featureSize,
        weights,
        hiddenSize,
        epsilon,
      });
      const gpuOutput = await runAudioSubsampleProjectionWebGpu(device, {
        inputFeatures: preprocessed.inputFeatures,
        inputFeaturesMask: preprocessed.inputFeaturesMask,
        frames: preprocessed.frames,
        featureSize: preprocessed.featureSize,
        weights,
        hiddenSize,
        epsilon,
      });
      const maxAbsDiff = maxAbsDifference(cpu.output, gpuOutput.output);
      return {
        ok: maxAbsDiff <= tolerance,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        source: { samples: samples.length, samplingRate },
        tensors: {
          layer0Conv: tensorSummaryForProbe(layer0Conv),
          layer0Norm: tensorSummaryForProbe(layer0Norm),
          layer1Conv: tensorSummaryForProbe(layer1Conv),
          layer1Norm: tensorSummaryForProbe(layer1Norm),
          inputProjection: tensorSummaryForProbe(inputProjection),
        },
        gpu: {
          adapter: true,
          device: true,
          frames: preprocessed.frames,
          featureSize: preprocessed.featureSize,
          outputRows: cpu.rows,
          validRows: cpu.validRows,
          hiddenSize,
          tolerance,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff,
          cpuChecksum: checksumFloats(cpu.output),
          gpuChecksum: checksumFloats(gpuOutput.output),
          firstCpuValues: roundedSample(cpu.output),
          firstGpuValues: roundedSample(gpuOutput.output),
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
      source: { samples: samples.length, samplingRate },
      tensors: {
        layer0Conv: null,
        layer0Norm: null,
        layer1Conv: null,
        layer1Norm: null,
        inputProjection: null,
      },
      gpu: {
        adapter: false,
        device: false,
        frames: 0,
        featureSize: 0,
        outputRows: 0,
        validRows: 0,
        hiddenSize: 0,
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

export async function probeGemma4AudioFeedForwardKernel(
  layerIndex = 0,
  block: Gemma4AudioFeedForwardBlockName = "feed_forward1",
  signal?: AbortSignal,
): Promise<Gemma4AudioFeedForwardKernelProbe> {
  const started = performance.now();
  const checkedLayerIndex = checkedAudioLayerIndex(layerIndex);
  const rows = 1;
  const hiddenSize = 1024;
  const intermediateSize = hiddenSize * 4;
  const epsilon = 0.000001;
  const residualWeight = AUDIO_FEED_FORWARD_RESIDUAL_WEIGHT;
  const tolerance = 0.01;

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
      const normNames = audioFeedForwardNormTensors(checkedLayerIndex, block);
      const [preLayerNormData, postLayerNormData] = await Promise.all([
        loadSafetensorsTensorData(urls.safetensors, header, normNames.preLayerNorm, signal),
        loadSafetensorsTensorData(urls.safetensors, header, normNames.postLayerNorm, signal),
      ]);
      assertF32Tensor(preLayerNormData, [hiddenSize], `audio layer ${checkedLayerIndex} ${block} pre layernorm`);
      assertF32Tensor(postLayerNormData, [hiddenSize], `audio layer ${checkedLayerIndex} ${block} post layernorm`);

      const hiddenStates = deterministicMultimodalHiddenStates(rows, hiddenSize);
      const feedForward = await computeAudioFeedForward({
        device,
        urls,
        header,
        layerIndex: checkedLayerIndex,
        block,
        hiddenStates,
        rows,
        hiddenSize,
        intermediateSize,
        epsilon,
        residualWeight,
        signal,
      });
      const maxAbsDiff = maxAbsDifference(feedForward.cpuOutput, feedForward.gpuOutput);

      return {
        ok: maxAbsDiff <= tolerance,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        layerIndex: checkedLayerIndex,
        block,
        tensors: {
          preLayerNorm: tensorSummaryForProbe(preLayerNormData),
          postLayerNorm: tensorSummaryForProbe(postLayerNormData),
          projections: feedForward.projections.map(audioFeedForwardProjectionSummary),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          intermediateSize,
          activation: "silu",
          residualWeight,
          epsilon,
          tolerance,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff,
          cpuChecksum: checksumFloats(feedForward.cpuOutput),
          gpuChecksum: checksumFloats(feedForward.gpuOutput),
          firstCpuValues: roundedSample(feedForward.cpuOutput),
          firstGpuValues: roundedSample(feedForward.gpuOutput),
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
      layerIndex,
      block,
      tensors: {
        preLayerNorm: null,
        postLayerNorm: null,
        projections: [],
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        intermediateSize: 0,
        activation: "silu",
        residualWeight,
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

export async function probeGemma4AudioLightConvKernel(
  layerIndex = 0,
  signal?: AbortSignal,
): Promise<Gemma4AudioLightConvKernelProbe> {
  const started = performance.now();
  const checkedLayerIndex = checkedAudioLayerIndex(layerIndex);
  const rows = 4;
  const hiddenSize = 1024;
  const expandedSize = hiddenSize * 2;
  const kernelSize = 5;
  const epsilon = 0.000001;
  const tolerance = 0.01;

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
      const tensorNames = audioLightConvTensors(checkedLayerIndex);
      const [preLayerNormData, convNormData, depthwiseConvData] = await Promise.all([
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.preLayerNorm, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.convNorm, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.depthwiseConv, signal),
      ]);
      assertF32Tensor(preLayerNormData, [hiddenSize], `audio layer ${checkedLayerIndex} lconv pre layernorm`);
      assertF32Tensor(convNormData, [hiddenSize], `audio layer ${checkedLayerIndex} lconv norm`);
      assertF32Tensor(
        depthwiseConvData,
        [hiddenSize, 1, kernelSize],
        `audio layer ${checkedLayerIndex} lconv depthwise conv`,
      );

      const hiddenStates = deterministicMultimodalHiddenStates(rows, hiddenSize);
      const lightConv = await computeAudioLightConv({
        device,
        urls,
        header,
        layerIndex: checkedLayerIndex,
        hiddenStates,
        rows,
        hiddenSize,
        expandedSize,
        kernelSize,
        epsilon,
        signal,
      });
      const maxAbsDiff = maxAbsDifference(lightConv.cpuOutput, lightConv.gpuOutput);

      return {
        ok: maxAbsDiff <= tolerance,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        layerIndex: checkedLayerIndex,
        tensors: {
          preLayerNorm: tensorSummaryForProbe(preLayerNormData),
          convNorm: tensorSummaryForProbe(convNormData),
          depthwiseConv: tensorSummaryForProbe(depthwiseConvData),
          projections: lightConv.projections.map(audioLightConvProjectionSummary),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          expandedSize,
          kernelSize,
          activation: "glu+silu",
          epsilon,
          tolerance,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff,
          cpuChecksum: checksumFloats(lightConv.cpuOutput),
          gpuChecksum: checksumFloats(lightConv.gpuOutput),
          firstCpuValues: roundedSample(lightConv.cpuOutput),
          firstGpuValues: roundedSample(lightConv.gpuOutput),
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
      layerIndex,
      tensors: {
        preLayerNorm: null,
        convNorm: null,
        depthwiseConv: null,
        projections: [],
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        expandedSize: 0,
        kernelSize,
        activation: "glu+silu",
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

export async function probeGemma4AudioSelfAttentionKernel(
  layerIndex = 0,
  signal?: AbortSignal,
): Promise<Gemma4AudioSelfAttentionKernelProbe> {
  const started = performance.now();
  const checkedLayerIndex = checkedAudioLayerIndex(layerIndex);
  const rows = 4;
  const hiddenSize = 1024;
  const heads = 8;
  const headDim = 128;
  const chunkSize = 12;
  const contextLeft = 13;
  const contextRight = 0;
  const contextSize = chunkSize + contextLeft - 1 + contextRight;
  const positionLength = Math.floor(contextSize / 2) + 1;
  const softcap = 50;
  const tolerance = 0.01;

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
      const tensorNames = audioSelfAttentionTensors(checkedLayerIndex);
      const [perDimScaleData, relativeKeyProjectionData] = await Promise.all([
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.perDimScale, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.relativeKeyProjection, signal),
      ]);
      assertF32Tensor(perDimScaleData, [headDim], `audio layer ${checkedLayerIndex} attention per-dim scale`);
      assertF32Tensor(
        relativeKeyProjectionData,
        [hiddenSize, hiddenSize],
        `audio layer ${checkedLayerIndex} relative key projection`,
      );

      const hiddenStates = deterministicMultimodalHiddenStates(rows, hiddenSize);
      const attention = await computeAudioSelfAttention({
        device,
        urls,
        header,
        layerIndex: checkedLayerIndex,
        hiddenStates,
        rows,
        hiddenSize,
        heads,
        headDim,
        chunkSize,
        contextSize,
        positionLength,
        softcap,
        signal,
      });
      const maxAbsDiff = maxAbsDifference(attention.cpuOutput, attention.gpuOutput);

      return {
        ok: maxAbsDiff <= tolerance,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        layerIndex: checkedLayerIndex,
        tensors: {
          perDimScale: tensorSummaryForProbe(perDimScaleData),
          relativeKeyProjection: tensorSummaryForProbe(relativeKeyProjectionData),
          projections: attention.projections.map(audioSelfAttentionProjectionSummary),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          heads,
          headDim,
          chunkSize,
          contextSize,
          positionLength,
          softcap,
          tolerance,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff,
          cpuChecksum: checksumFloats(attention.cpuOutput),
          gpuChecksum: checksumFloats(attention.gpuOutput),
          firstCpuValues: roundedSample(attention.cpuOutput),
          firstGpuValues: roundedSample(attention.gpuOutput),
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
      layerIndex,
      tensors: {
        perDimScale: null,
        relativeKeyProjection: null,
        projections: [],
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        heads: 0,
        headDim: 0,
        chunkSize,
        contextSize,
        positionLength,
        softcap,
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

export async function probeGemma4AudioEncoderLayerKernel(
  layerIndex = 0,
  signal?: AbortSignal,
): Promise<Gemma4AudioEncoderLayerKernelProbe> {
  const started = performance.now();
  const checkedLayerIndex = checkedAudioLayerIndex(layerIndex);
  const rows = 1;
  const hiddenSize = 1024;
  const intermediateSize = hiddenSize * 4;
  const expandedSize = hiddenSize * 2;
  const heads = 8;
  const headDim = 128;
  const chunkSize = 12;
  const contextLeft = 13;
  const contextRight = 0;
  const contextSize = chunkSize + contextLeft - 1 + contextRight;
  const positionLength = Math.floor(contextSize / 2) + 1;
  const kernelSize = 5;
  const softcap = 50;
  const residualWeight = 0.5;
  const epsilon = 0.000001;
  const tolerance = 0.02;

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
      const tensorNames = audioEncoderLayerNormTensors(checkedLayerIndex);
      const [normPreAttentionData, normPostAttentionData, normOutData] = await Promise.all([
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.normPreAttention, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.normPostAttention, signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.normOut, signal),
      ]);
      assertF32Tensor(normPreAttentionData, [hiddenSize], `audio layer ${checkedLayerIndex} pre-attention norm`);
      assertF32Tensor(normPostAttentionData, [hiddenSize], `audio layer ${checkedLayerIndex} post-attention norm`);
      assertF32Tensor(normOutData, [hiddenSize], `audio layer ${checkedLayerIndex} output norm`);

      const hiddenStates = deterministicMultimodalHiddenStates(rows, hiddenSize);
      const layer = await computeAudioEncoderLayer({
        device,
        urls,
        header,
        layerIndex: checkedLayerIndex,
        hiddenStates,
        rows,
        hiddenSize,
        intermediateSize,
        expandedSize,
        heads,
        headDim,
        chunkSize,
        contextSize,
        positionLength,
        kernelSize,
        softcap,
        residualWeight,
        epsilon,
        signal,
      });
      const maxAbsDiff = maxAbsDifference(layer.cpuOutput, layer.gpuOutput);

      return {
        ok: maxAbsDiff <= tolerance,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        layerIndex: checkedLayerIndex,
        tensors: {
          normPreAttention: tensorSummaryForProbe(normPreAttentionData),
          normPostAttention: tensorSummaryForProbe(normPostAttentionData),
          normOut: tensorSummaryForProbe(normOutData),
          feedForward1: layer.feedForward1.projections.map(audioFeedForwardProjectionSummary),
          selfAttention: layer.selfAttention.projections.map(audioSelfAttentionProjectionSummary),
          lightConv: layer.lightConv.projections.map(audioLightConvProjectionSummary),
          feedForward2: layer.feedForward2.projections.map(audioFeedForwardProjectionSummary),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          intermediateSize,
          expandedSize,
          heads,
          headDim,
          chunkSize,
          contextSize,
          positionLength,
          kernelSize,
          residualWeight,
          epsilon,
          tolerance,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff,
          stageMaxAbsDiffs: layer.stageMaxAbsDiffs,
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
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      layerIndex,
      tensors: {
        normPreAttention: null,
        normPostAttention: null,
        normOut: null,
        feedForward1: [],
        selfAttention: [],
        lightConv: [],
        feedForward2: [],
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        intermediateSize: 0,
        expandedSize: 0,
        heads: 0,
        headDim: 0,
        chunkSize,
        contextSize,
        positionLength,
        kernelSize,
        residualWeight,
        epsilon,
        tolerance,
        workgroupSize: 64,
      },
      comparison: {
        maxAbsDiff: null,
        stageMaxAbsDiffs: {},
        cpuChecksum: null,
        gpuChecksum: null,
        firstCpuValues: [],
        firstGpuValues: [],
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function probeGemma4AudioEncoderStackKernel(
  layerCount = AUDIO_ENCODER_LAYER_COUNT,
  signal?: AbortSignal,
): Promise<Gemma4AudioEncoderStackKernelProbe> {
  const started = performance.now();
  const checkedLayerCount = checkedAudioLayerCount(layerCount);
  const rows = 1;
  const hiddenSize = 1024;
  const intermediateSize = hiddenSize * 4;
  const expandedSize = hiddenSize * 2;
  const heads = 8;
  const headDim = 128;
  const chunkSize = 12;
  const contextLeft = 13;
  const contextRight = 0;
  const contextSize = chunkSize + contextLeft - 1 + contextRight;
  const positionLength = Math.floor(contextSize / 2) + 1;
  const kernelSize = 5;
  const softcap = 50;
  const residualWeight = AUDIO_FEED_FORWARD_RESIDUAL_WEIGHT;
  const epsilon = 0.000001;
  const tolerance = 0.05;

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
      const stack = await computeAudioEncoderStack({
        device,
        urls,
        header,
        layerCount: checkedLayerCount,
        hiddenStates: deterministicMultimodalHiddenStates(rows, hiddenSize),
        rows,
        hiddenSize,
        intermediateSize,
        expandedSize,
        heads,
        headDim,
        chunkSize,
        contextSize,
        positionLength,
        kernelSize,
        softcap,
        residualWeight,
        epsilon,
        signal,
      });

      return {
        ok: stack.maxAbsDiff <= tolerance,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        layerCount: checkedLayerCount,
        layers: stack.layers,
        gpuPath: {
          completed: true,
          finite: allFinite(stack.gpuOutput),
          checksum: checksumFloats(stack.gpuOutput),
          firstValues: roundedSample(stack.gpuOutput),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          intermediateSize,
          expandedSize,
          heads,
          headDim,
          chunkSize,
          contextSize,
          positionLength,
          kernelSize,
          residualWeight,
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
      layers: [],
      gpuPath: {
        completed: false,
        finite: false,
        checksum: null,
        firstValues: [],
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        intermediateSize: 0,
        expandedSize: 0,
        heads: 0,
        headDim: 0,
        chunkSize,
        contextSize,
        positionLength,
        kernelSize,
        residualWeight,
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

export async function probeGemma4AudioEncoderStackStepKernel(
  options: { layersPerStep?: number; reset?: boolean } = {},
  signal?: AbortSignal,
): Promise<Gemma4AudioEncoderStackStepKernelProbe> {
  const started = performance.now();
  const rows = 1;
  const hiddenSize = 1024;
  const intermediateSize = hiddenSize * 4;
  const expandedSize = hiddenSize * 2;
  const heads = 8;
  const headDim = 128;
  const chunkSize = 12;
  const contextLeft = 13;
  const contextRight = 0;
  const contextSize = chunkSize + contextLeft - 1 + contextRight;
  const positionLength = Math.floor(contextSize / 2) + 1;
  const kernelSize = 5;
  const softcap = 50;
  const residualWeight = AUDIO_FEED_FORWARD_RESIDUAL_WEIGHT;
  const epsilon = 0.000001;
  const tolerance = 0.05;
  const requestedLayersPerStep = options.layersPerStep ?? 2;
  const parsedLayersPerStep = Number.isFinite(requestedLayersPerStep) ? Math.floor(requestedLayersPerStep) : 2;
  const layersPerStep = Math.max(1, Math.min(AUDIO_ENCODER_LAYER_COUNT, parsedLayersPerStep));

  try {
    if (options.reset || !audioEncoderStackStepSession || audioEncoderStackStepSession.nextLayerIndex >= AUDIO_ENCODER_LAYER_COUNT) {
      const hiddenStates = deterministicMultimodalHiddenStates(rows, hiddenSize);
      audioEncoderStackStepSession = {
        id: nextAudioEncoderStackStepSessionId,
        cpuHiddenStates: hiddenStates,
        gpuHiddenStates: hiddenStates,
        nextLayerIndex: 0,
        layers: [],
      };
      nextAudioEncoderStackStepSessionId += 1;
    }

    const session = audioEncoderStackStepSession;
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
      const endLayer = Math.min(AUDIO_ENCODER_LAYER_COUNT, session.nextLayerIndex + layersPerStep);
      for (let layerIndex = session.nextLayerIndex; layerIndex < endLayer; layerIndex += 1) {
        const layer = await computeAudioEncoderLayer({
          device,
          urls,
          header,
          layerIndex,
          hiddenStates: session.cpuHiddenStates,
          gpuHiddenStates: session.gpuHiddenStates,
          rows,
          hiddenSize,
          intermediateSize,
          expandedSize,
          heads,
          headDim,
          chunkSize,
          contextSize,
          positionLength,
          kernelSize,
          softcap,
          residualWeight,
          epsilon,
          signal,
        });
        session.cpuHiddenStates = layer.cpuOutput;
        session.gpuHiddenStates = layer.gpuOutput;
        const maxAbsDiff = maxAbsDifference(session.cpuHiddenStates, session.gpuHiddenStates);
        session.layers.push({
          layerIndex,
          maxAbsDiff,
          stageMaxAbsDiffs: layer.stageMaxAbsDiffs,
          cpuChecksum: checksumFloats(session.cpuHiddenStates),
          gpuChecksum: checksumFloats(session.gpuHiddenStates),
          firstCpuValues: roundedSample(session.cpuHiddenStates),
          firstGpuValues: roundedSample(session.gpuHiddenStates),
        });
        session.nextLayerIndex = layerIndex + 1;
      }
      const finalMaxAbsDiff = maxAbsDifference(session.cpuHiddenStates, session.gpuHiddenStates);
      return {
        ok: finalMaxAbsDiff <= tolerance,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        sessionId: session.id,
        completed: session.nextLayerIndex >= AUDIO_ENCODER_LAYER_COUNT,
        processedLayers: session.layers.length,
        nextLayerIndex: session.nextLayerIndex,
        totalLayers: AUDIO_ENCODER_LAYER_COUNT,
        layers: session.layers,
        gpuPath: {
          completed: session.nextLayerIndex >= AUDIO_ENCODER_LAYER_COUNT,
          finite: allFinite(session.gpuHiddenStates),
          checksum: checksumFloats(session.gpuHiddenStates),
          firstValues: roundedSample(session.gpuHiddenStates),
        },
        gpu: {
          adapter: true,
          device: true,
          rows,
          hiddenSize,
          intermediateSize,
          expandedSize,
          heads,
          headDim,
          chunkSize,
          contextSize,
          positionLength,
          kernelSize,
          residualWeight,
          epsilon,
          tolerance,
          workgroupSize: 64,
        },
        comparison: {
          maxAbsDiff: finalMaxAbsDiff,
          cpuChecksum: checksumFloats(session.cpuHiddenStates),
          gpuChecksum: checksumFloats(session.gpuHiddenStates),
          firstCpuValues: roundedSample(session.cpuHiddenStates),
          firstGpuValues: roundedSample(session.gpuHiddenStates),
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
      sessionId: audioEncoderStackStepSession?.id ?? null,
      completed: false,
      processedLayers: audioEncoderStackStepSession?.layers.length ?? 0,
      nextLayerIndex: audioEncoderStackStepSession?.nextLayerIndex ?? 0,
      totalLayers: AUDIO_ENCODER_LAYER_COUNT,
      layers: audioEncoderStackStepSession?.layers ?? [],
      gpuPath: {
        completed: false,
        finite: audioEncoderStackStepSession ? allFinite(audioEncoderStackStepSession.gpuHiddenStates) : false,
        checksum: audioEncoderStackStepSession ? checksumFloats(audioEncoderStackStepSession.gpuHiddenStates) : null,
        firstValues: audioEncoderStackStepSession ? roundedSample(audioEncoderStackStepSession.gpuHiddenStates) : [],
      },
      gpu: {
        adapter: false,
        device: false,
        rows: 0,
        hiddenSize: 0,
        intermediateSize: 0,
        expandedSize: 0,
        heads: 0,
        headDim: 0,
        chunkSize,
        contextSize,
        positionLength,
        kernelSize,
        residualWeight,
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
