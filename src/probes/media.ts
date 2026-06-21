import { navigatorGpu } from "../gpu/runtime.js";
import { fetchJson, fetchSafetensorsHeader, float32FromBytes, loadSafetensorsTensorData, loadSafetensorsTensorSlice, numberOrNull, record } from "../io/safetensors.js";
import { runProjectionCpu, runProjectionWebGpu } from "../kernels/qat.js";
import { GEMMA4_MEDIA_MODEL_ID, HF_RESOLVE_BASE, MEDIA_TENSOR_CACHE_NAME } from "../model.js";
import { computeGemma4AudioSoftTokenCount } from "../preprocess/audio.js";
import { computeGemma4ImageTokenLayout } from "../preprocess/image.js";
import { Gemma4MediaKernelProbe, Gemma4MediaProjectionKernelProbe, Gemma4MediaTensorSliceProbe, JsonRecord } from "../types.js";
import { checksumBytes, checksumFloats, deterministicProjectionInput, maxAbsDifference, roundedSample } from "../utils/math.js";

export async function probeGemma4MediaKernelArtifact(
  signal?: AbortSignal,
): Promise<Gemma4MediaKernelProbe> {
  const started = performance.now();
  const urls = mediaArtifactUrls();
  try {
    const [config, processorConfig, preprocessorConfig, safetensorsHeader] = await Promise.all([
      fetchJson(urls.config, signal),
      fetchJson(urls.processorConfig, signal),
      fetchJson(urls.preprocessorConfig, signal),
      fetchSafetensorsHeader(urls.safetensors, signal),
    ]);

    const processor = mediaProcessorSummary(processorConfig, preprocessorConfig);
    const tensorNames = Object.keys(safetensorsHeader.tensors);
    const tensors = tensorSummary(tensorNames, safetensorsHeader.headerBytes);
    const imagePatchSize = processor.imagePatchSize ?? 16;
    const imagePoolingKernelSize = processor.imagePoolingKernelSize ?? 3;
    const imageMaxSoftTokens = processor.imageMaxSoftTokens ?? 280;

    return {
      ok: tensors.visionTower > 0 &&
        tensors.audioTower > 0 &&
        tensors.embedVision > 0 &&
        tensors.embedAudio > 0,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      urls,
      config: mediaConfigSummary(config),
      processor,
      referenceLayouts: {
        image: computeGemma4ImageTokenLayout({
          height: 1024,
          width: 768,
          patchSize: imagePatchSize,
          maxSoftTokens: imageMaxSoftTokens,
          poolingKernelSize: imagePoolingKernelSize,
        }),
        audioSoftTokensForFiveSeconds: computeGemma4AudioSoftTokenCount({
          samples: 5 * (processor.audioSamplingRate ?? 16_000),
          frameLength: processor.audioFrameLength ?? 320,
          hopLength: processor.audioHopLength ?? 160,
          audioSeqLength: processor.audioSeqLength ?? 750,
        }),
      },
      tensors,
      implementationStages: [
        "wire browser image preprocessor outputs into the WebGPU vision_tower input path",
        "wire browser audio preprocessor outputs into the WebGPU audio_tower input path",
        "wire cached safetensors tensor slices into the media WebGPU kernels",
        "implement WebGPU kernels for vision_tower, audio_tower, embed_vision, and embed_audio",
        "extend the text kernel entry point to accept multimodal placeholders plus soft-token embeddings",
        "verify image understanding and voice-note transcription fixtures in Orca before enabling product capabilities",
      ],
      capabilitiesReady: false,
    };
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      urls,
      config: emptyConfigSummary(),
      processor: emptyProcessorSummary(),
      referenceLayouts: {
        image: null,
        audioSoftTokensForFiveSeconds: null,
      },
      tensors: {
        headerBytes: 0,
        total: 0,
        visionTower: 0,
        audioTower: 0,
        embedVision: 0,
        embedAudio: 0,
        samples: {
          visionTower: [],
          audioTower: [],
          embedVision: [],
          embedAudio: [],
        },
      },
      implementationStages: [],
      capabilitiesReady: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function probeGemma4MediaTensorSlices(
  signal?: AbortSignal,
): Promise<Gemma4MediaTensorSliceProbe> {
  const started = performance.now();
  const urls = mediaArtifactUrls();
  const tensorNames = [
    "model.embed_vision.embedding_projection.weight",
    "model.embed_audio.embedding_projection.weight",
    "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.input_activation_scale",
    "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.input_activation_scale",
  ];
  try {
    const header = await fetchSafetensorsHeader(urls.safetensors, signal);
    const tensors = await Promise.all(
      tensorNames.map((name) => loadSafetensorsTensorSlice(urls.safetensors, header, name, signal)),
    );
    return {
      ok: tensors.every((tensor) => tensor.dataBytes > 0),
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      tensors,
      totalBytes: tensors.reduce((sum, tensor) => sum + tensor.dataBytes, 0),
      allFromCache: tensors.every((tensor) => tensor.fromCache),
      cache: `Cache API:${MEDIA_TENSOR_CACHE_NAME}`,
    };
  } catch (error) {
    return {
      ok: false,
      modelId: GEMMA4_MEDIA_MODEL_ID,
      durationMs: Math.round(performance.now() - started),
      tensors: [],
      totalBytes: 0,
      allFromCache: false,
      cache: `Cache API:${MEDIA_TENSOR_CACHE_NAME}`,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function probeGemma4MediaProjectionKernel(
  kind: "vision" | "audio" = "vision",
  signal?: AbortSignal,
): Promise<Gemma4MediaProjectionKernelProbe> {
  const started = performance.now();
  const tensorName = kind === "vision"
    ? "model.embed_vision.embedding_projection.weight"
    : "model.embed_audio.embedding_projection.weight";
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
      const tensorData = await loadSafetensorsTensorData(urls.safetensors, header, tensorName, signal);
      if (tensorData.dtype !== "F32") {
        throw new Error(`Expected F32 projection tensor, got ${tensorData.dtype}.`);
      }
      if (tensorData.shape.length !== 2) {
        throw new Error(`Expected 2-D projection tensor, got shape [${tensorData.shape.join(", ")}].`);
      }

      const [outputDim, inputDim] = tensorData.shape;
      const weights = float32FromBytes(tensorData.bytes);
      const input = deterministicProjectionInput(inputDim);
      const cpuOutput = runProjectionCpu(input, weights, inputDim, outputDim);
      const gpuOutput = await runProjectionWebGpu(device, input, weights, inputDim, outputDim);
      const maxAbsDiff = maxAbsDifference(cpuOutput, gpuOutput);

      return {
        ok: maxAbsDiff <= 0.001,
        modelId: GEMMA4_MEDIA_MODEL_ID,
        durationMs: Math.round(performance.now() - started),
        kind,
        tensor: {
          name: tensorData.name,
          dtype: tensorData.dtype,
          shape: tensorData.shape,
          dataBytes: tensorData.dataBytes,
          fromCache: tensorData.fromCache,
          checksum: checksumBytes(tensorData.bytes),
        },
        gpu: {
          adapter: true,
          device: true,
          inputDim,
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
      kind,
      tensor: null,
      gpu: {
        adapter: false,
        device: false,
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

export function mediaArtifactUrls(): Gemma4MediaKernelProbe["urls"] {
  const base = `${HF_RESOLVE_BASE}/${GEMMA4_MEDIA_MODEL_ID}/resolve/main`;
  return {
    config: `${base}/config.json`,
    processorConfig: `${base}/processor_config.json`,
    preprocessorConfig: `${base}/preprocessor_config.json`,
    safetensors: `${base}/model.safetensors`,
  };
}

export function mediaConfigSummary(config: JsonRecord): Gemma4MediaKernelProbe["config"] {
  const textConfig = record(config.text_config);
  const visionConfig = record(config.vision_config);
  const audioConfig = record(config.audio_config);
  return {
    imageTokenId: numberOrNull(config.image_token_id),
    audioTokenId: numberOrNull(config.audio_token_id),
    videoTokenId: numberOrNull(config.video_token_id),
    visionSoftTokensPerImage: numberOrNull(config.vision_soft_tokens_per_image),
    textHiddenSize: numberOrNull(textConfig.hidden_size),
    textLayers: numberOrNull(textConfig.num_hidden_layers),
    visionHiddenSize: numberOrNull(visionConfig.hidden_size),
    visionLayers: numberOrNull(visionConfig.num_hidden_layers),
    audioHiddenSize: numberOrNull(audioConfig.hidden_size),
    audioLayers: numberOrNull(audioConfig.num_hidden_layers),
  };
}

export function mediaProcessorSummary(
  processorConfig: JsonRecord,
  preprocessorConfig: JsonRecord,
): Gemma4MediaKernelProbe["processor"] {
  const imageProcessor = record(processorConfig.image_processor);
  const featureExtractor = record(processorConfig.feature_extractor);
  return {
    imageSeqLength: numberOrNull(processorConfig.image_seq_length),
    imagePatchSize: numberOrNull(imageProcessor.patch_size),
    imageMaxSoftTokens: numberOrNull(imageProcessor.max_soft_tokens),
    imagePoolingKernelSize: numberOrNull(imageProcessor.pooling_kernel_size),
    imageRescaleFactor: numberOrNull(imageProcessor.rescale_factor),
    audioSeqLength: numberOrNull(processorConfig.audio_seq_length),
    audioMsPerToken: numberOrNull(processorConfig.audio_ms_per_token),
    audioSamplingRate: numberOrNull(featureExtractor.sampling_rate ?? preprocessorConfig.sampling_rate),
    audioFeatureSize: numberOrNull(featureExtractor.feature_size ?? preprocessorConfig.feature_size),
    audioFrameLength: numberOrNull(featureExtractor.frame_length ?? preprocessorConfig.frame_length),
    audioHopLength: numberOrNull(featureExtractor.hop_length ?? preprocessorConfig.hop_length),
    audioFftLength: numberOrNull(featureExtractor.fft_length ?? preprocessorConfig.fft_length),
  };
}

export function emptyConfigSummary(): Gemma4MediaKernelProbe["config"] {
  return {
    imageTokenId: null,
    audioTokenId: null,
    videoTokenId: null,
    visionSoftTokensPerImage: null,
    textHiddenSize: null,
    textLayers: null,
    visionHiddenSize: null,
    visionLayers: null,
    audioHiddenSize: null,
    audioLayers: null,
  };
}

export function emptyProcessorSummary(): Gemma4MediaKernelProbe["processor"] {
  return {
    imageSeqLength: null,
    imagePatchSize: null,
    imageMaxSoftTokens: null,
    imagePoolingKernelSize: null,
    imageRescaleFactor: null,
    audioSeqLength: null,
    audioMsPerToken: null,
    audioSamplingRate: null,
    audioFeatureSize: null,
    audioFrameLength: null,
    audioHopLength: null,
    audioFftLength: null,
  };
}

export function tensorSummary(names: string[], headerBytes: number): Gemma4MediaKernelProbe["tensors"] {
  const visionTower = names.filter((name) => name.startsWith("model.vision_tower."));
  const audioTower = names.filter((name) => name.startsWith("model.audio_tower."));
  const embedVision = names.filter((name) => name.startsWith("model.embed_vision."));
  const embedAudio = names.filter((name) => name.startsWith("model.embed_audio."));
  return {
    headerBytes,
    total: names.length,
    visionTower: visionTower.length,
    audioTower: audioTower.length,
    embedVision: embedVision.length,
    embedAudio: embedAudio.length,
    samples: {
      visionTower: visionTower.slice(0, 5),
      audioTower: audioTower.slice(0, 5),
      embedVision: embedVision.slice(0, 5),
      embedAudio: embedAudio.slice(0, 5),
    },
  };
}