import { computeAudioEncoderStackGpuOnly, stripRowsByMask } from "./encoder.js";
import { runAudioOutputProjectionWebGpu, runAudioSubsampleProjectionWebGpu } from "./kernels.js";
import { computeMultimodalEmbedderWebGpuOnly } from "../embedder/index.js";
import { navigatorGpu } from "../gpu/runtime.js";
import { assertF32Tensor, fetchSafetensorsHeader, float32FromBytes, loadSafetensorsTensorData } from "../io/safetensors.js";
import { AUDIO_ENCODER_LAYER_COUNT, AUDIO_FEED_FORWARD_RESIDUAL_WEIGHT, GEMMA4_MEDIA_MODEL_ID, audioSubsampleTensorNames, checkedAudioLayerCount } from "../model.js";
import { preprocessGemma4AudioSamples } from "../preprocess/audio.js";
import { mediaArtifactUrls } from "../probes/media.js";
import { Gemma4AudioEmbeddingResult } from "../types.js";
import { allFinite, checksumFloats, roundedSample } from "../utils/math.js";

export async function computeGemma4AudioEmbeddings(input: {
  samples: Float32Array | number[];
  samplingRate?: number;
  layerCount?: number;
  maxSamples?: number;
  signal?: AbortSignal;
}): Promise<Gemma4AudioEmbeddingResult> {
  const started = performance.now();
  const samplingRate = input.samplingRate ?? 16_000;
  const hiddenSize = 1024;
  const intermediateSize = hiddenSize * 4;
  const expandedSize = hiddenSize * 2;
  const audioOutputDim = 1536;
  const textHiddenSize = 1536;
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
  const layerCount = input.layerCount ?? AUDIO_ENCODER_LAYER_COUNT;

  try {
    const checkedLayerCount = checkedAudioLayerCount(layerCount);
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
      const preprocessed = preprocessGemma4AudioSamples(input.samples, {
        samplingRate,
        maxSamples: input.maxSamples,
        padToMultipleOf: null,
      });
      if (preprocessed.frames === 0) {
        throw new Error("Audio preprocessing produced no frames.");
      }
      const tensorNames = audioSubsampleTensorNames();
      const [layer0Conv, layer0Norm, layer1Conv, layer1Norm, inputProjection] = await Promise.all([
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.layer0Conv, input.signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.layer0Norm, input.signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.layer1Conv, input.signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.layer1Norm, input.signal),
        loadSafetensorsTensorData(urls.safetensors, header, tensorNames.inputProjection, input.signal),
      ]);
      assertF32Tensor(layer0Conv, [128, 1, 3, 3], "audio embedding subsample layer0 conv");
      assertF32Tensor(layer0Norm, [128], "audio embedding subsample layer0 norm");
      assertF32Tensor(layer1Conv, [32, 128, 3, 3], "audio embedding subsample layer1 conv");
      assertF32Tensor(layer1Norm, [32], "audio embedding subsample layer1 norm");
      assertF32Tensor(inputProjection, [hiddenSize, hiddenSize], "audio embedding subsample input projection");

      const subsample = await runAudioSubsampleProjectionWebGpu(device, {
        inputFeatures: preprocessed.inputFeatures,
        inputFeaturesMask: preprocessed.inputFeaturesMask,
        frames: preprocessed.frames,
        featureSize: preprocessed.featureSize,
        weights: {
          layer0Conv: float32FromBytes(layer0Conv.bytes),
          layer0Norm: float32FromBytes(layer0Norm.bytes),
          layer1Conv: float32FromBytes(layer1Conv.bytes),
          layer1Norm: float32FromBytes(layer1Norm.bytes),
          inputProjection: float32FromBytes(inputProjection.bytes),
        },
        hiddenSize,
        epsilon,
      });
      const stackGpuOutput = await computeAudioEncoderStackGpuOnly({
        device,
        urls,
        header,
        layerCount: checkedLayerCount,
        hiddenStates: subsample.output,
        rows: subsample.rows,
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
        signal: input.signal,
      });
      const outputProjection = await runAudioOutputProjectionWebGpu({
        device,
        urls,
        header,
        hiddenStates: stackGpuOutput,
        rows: subsample.rows,
        inputDim: hiddenSize,
        outputDim: audioOutputDim,
        signal: input.signal,
      });
      const projectedRows = stripRowsByMask(outputProjection, subsample.mask, audioOutputDim);
      const embeddings = await computeMultimodalEmbedderWebGpuOnly({
        device,
        urls,
        header,
        kind: "audio",
        hiddenStates: projectedRows,
        rows: subsample.validRows,
        inputDim: audioOutputDim,
        outputDim: textHiddenSize,
        epsilon,
        signal: input.signal,
      });
      const finite = allFinite(embeddings);
      return {
        ok: finite,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        source: { samples: Math.min(input.samples.length, input.maxSamples ?? input.samples.length), samplingRate },
        layerCount: checkedLayerCount,
        rows: subsample.validRows,
        dim: textHiddenSize,
        embeddings,
        finite,
        checksum: checksumFloats(embeddings),
        firstValues: roundedSample(embeddings),
        stages: {
          frames: preprocessed.frames,
          validFrames: preprocessed.validFrames,
          subsampleRows: subsample.rows,
          validSubsampleRows: subsample.validRows,
          stackFinite: allFinite(stackGpuOutput),
          outputProjectionFinite: allFinite(outputProjection),
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
      source: { samples: 0, samplingRate },
      layerCount,
      rows: 0,
      dim: 0,
      embeddings: new Float32Array(),
      finite: false,
      checksum: null,
      firstValues: [],
      stages: {
        frames: 0,
        validFrames: 0,
        subsampleRows: 0,
        validSubsampleRows: 0,
        stackFinite: false,
        outputProjectionFinite: false,
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}