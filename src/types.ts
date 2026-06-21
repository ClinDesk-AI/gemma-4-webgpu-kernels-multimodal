export interface QatProjectionTensorNames {
  weight: string;
  weightScale: string;
  inputScale: string;
  outputScale: string;
}

export interface VisionEncoderLayerNormTensorNames {
  inputLayerNorm: string;
  postAttentionLayerNorm: string;
  preFeedForwardLayerNorm: string;
  postFeedForwardLayerNorm: string;
}

export type JsonRecord = Record<string, unknown>;

export type Gemma4AudioFeedForwardBlockName = "feed_forward1" | "feed_forward2";

export type Gemma4AudioFeedForwardProjectionName = "ffw_layer_1" | "ffw_layer_2";

export type Gemma4AudioLightConvProjectionName = "linear_start" | "linear_end";

export type Gemma4AudioSelfAttentionProjectionName = "q_proj" | "k_proj" | "v_proj" | "post";

export interface SafetensorsTensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface SafetensorsHeader {
  headerBytes: number;
  tensors: Record<string, SafetensorsTensorInfo>;
}

export interface SafetensorsTensorData {
  name: string;
  dtype: string;
  shape: number[];
  dataBytes: number;
  absoluteStart: number;
  absoluteEndInclusive: number;
  fromCache: boolean;
  bytes: Uint8Array;
}

export interface SafetensorsTensorByteRange {
  name: string;
  dataBytes: number;
  absoluteStart: number;
  absoluteEndInclusive: number;
  fromCache: boolean;
  bytes: Uint8Array;
}

export interface Gemma4ImageTokenLayout {
  input: { height: number; width: number };
  target: { height: number; width: number };
  patchGrid: { height: number; width: number };
  maxPatches: number;
  softTokens: number;
}

export interface Gemma4MediaKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  urls: {
    config: string;
    processorConfig: string;
    preprocessorConfig: string;
    safetensors: string;
  };
  config: {
    imageTokenId: number | null;
    audioTokenId: number | null;
    videoTokenId: number | null;
    visionSoftTokensPerImage: number | null;
    textHiddenSize: number | null;
    textLayers: number | null;
    visionHiddenSize: number | null;
    visionLayers: number | null;
    audioHiddenSize: number | null;
    audioLayers: number | null;
  };
  processor: {
    imageSeqLength: number | null;
    imagePatchSize: number | null;
    imageMaxSoftTokens: number | null;
    imagePoolingKernelSize: number | null;
    imageRescaleFactor: number | null;
    audioSeqLength: number | null;
    audioMsPerToken: number | null;
    audioSamplingRate: number | null;
    audioFeatureSize: number | null;
    audioFrameLength: number | null;
    audioHopLength: number | null;
    audioFftLength: number | null;
  };
  referenceLayouts: {
    image: Gemma4ImageTokenLayout | null;
    audioSoftTokensForFiveSeconds: number | null;
  };
  tensors: {
    headerBytes: number;
    total: number;
    visionTower: number;
    audioTower: number;
    embedVision: number;
    embedAudio: number;
    samples: {
      visionTower: string[];
      audioTower: string[];
      embedVision: string[];
      embedAudio: string[];
    };
  };
  implementationStages: string[];
  capabilitiesReady: false;
  error?: string;
}

export interface Gemma4MediaTensorSliceProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  tensors: Array<{
    name: string;
    dtype: string;
    shape: number[];
    dataBytes: number;
    absoluteStart: number;
    absoluteEndInclusive: number;
    fromCache: boolean;
    checksum: number;
  }>;
  totalBytes: number;
  allFromCache: boolean;
  cache: string;
  error?: string;
}

export interface Gemma4MediaProjectionKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  kind: "vision" | "audio";
  tensor: {
    name: string;
    dtype: string;
    shape: number[];
    dataBytes: number;
    fromCache: boolean;
    checksum: number;
  } | null;
  gpu: {
    adapter: boolean;
    device: boolean;
    inputDim: number;
    outputDim: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4VisionPatchEmbeddingKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  tensors: {
    inputProjection: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
    positionRows: Array<{
      axis: "x" | "y";
      indexStart: number;
      rowCount: number;
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    }>;
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    patches: number;
    patchPixels: number;
    hiddenSize: number;
    outputRows: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4VisionRmsNormKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  tensor: {
    name: string;
    dtype: string;
    shape: number[];
    dataBytes: number;
    fromCache: boolean;
    checksum: number;
  } | null;
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    epsilon: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export type Gemma4VisionAttentionProjectionName = "q_proj" | "k_proj" | "v_proj";

export interface Gemma4VisionAttentionProjectionResult {
  projection: Gemma4VisionAttentionProjectionName;
  tensors: {
    weight: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
    weightScale: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
  };
  quantization: {
    inputActivationScale: number | null;
    outputActivationScale: number | null;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4VisionAttentionProjectionKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  projections: Gemma4VisionAttentionProjectionResult[];
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    inputDim: number;
    outputDim: number;
    workgroupSize: number;
  };
  error?: string;
}

export interface Gemma4VisionAttentionNormResult {
  projection: Gemma4VisionAttentionProjectionName;
  withScale: boolean;
  tensor: {
    name: string;
    dtype: string;
    shape: number[];
    dataBytes: number;
    fromCache: boolean;
    checksum: number;
  } | null;
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4VisionAttentionNormKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  norms: Gemma4VisionAttentionNormResult[];
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    heads: number;
    headDim: number;
    epsilon: number;
    workgroupSize: number;
  };
  error?: string;
}

export interface Gemma4VisionAttentionRopeResult {
  projection: "q_proj" | "k_proj";
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4VisionAttentionRopeKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  ropes: Gemma4VisionAttentionRopeResult[];
  positions: Array<{ x: number; y: number }>;
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    heads: number;
    headDim: number;
    ropeTheta: number;
    workgroupSize: number;
  };
  error?: string;
}

export interface Gemma4VisionAttentionBodyKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    heads: number;
    headDim: number;
    scaling: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4VisionAttentionOutputProjectionKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  tensors: {
    weight: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
    weightScale: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
  };
  quantization: {
    inputActivationScale: number | null;
    outputActivationScale: number | null;
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    inputDim: number;
    outputDim: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4VisionPostAttentionKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  tensors: {
    inputLayerNorm: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
    postAttentionLayerNorm: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    heads: number;
    headDim: number;
    epsilon: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export type Gemma4VisionMlpProjectionName = "gate_proj" | "up_proj" | "down_proj";

export interface Gemma4VisionMlpProjectionTensorSummary {
  projection: Gemma4VisionMlpProjectionName;
  weight: {
    name: string;
    dtype: string;
    shape: number[];
    dataBytes: number;
    fromCache: boolean;
    checksum: number;
  } | null;
  weightScale: {
    name: string;
    dtype: string;
    shape: number[];
    dataBytes: number;
    fromCache: boolean;
    checksum: number;
  } | null;
  inputActivationScale: number | null;
  outputActivationScale: number | null;
}

export interface Gemma4VisionFeedForwardKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  tensors: {
    preFeedForwardLayerNorm: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
    postFeedForwardLayerNorm: {
      name: string;
      dtype: string;
      shape: number[];
      dataBytes: number;
      fromCache: boolean;
      checksum: number;
    } | null;
    mlpProjections: Gemma4VisionMlpProjectionTensorSummary[];
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    intermediateSize: number;
    activation: "gelu_pytorch_tanh";
    epsilon: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export type Gemma4VisionEncoderLayerKernelProbe = Gemma4VisionFeedForwardKernelProbe & {
  layerIndex: number;
};

export interface Gemma4VisionEncoderStackKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  layerCount: number;
  comparisonMode: "propagated" | "gpuAnchored";
  layers: Array<{
    layerIndex: number;
    maxAbsDiff: number;
    maxAbsDiffIndex: number;
    diffCountAboveTolerance: number;
    attentionMaxAbsDiff: number | null;
    mlpMaxAbsDiff: number | null;
    attentionStageMaxAbsDiff: {
      qNorm: number;
      kNorm: number;
      vNorm: number;
      qRope: number;
      kRope: number;
      body: number;
      outputProjection: number;
    } | null;
    cpuChecksum: number;
    gpuChecksum: number;
    firstCpuValues: number[];
    firstGpuValues: number[];
    maxCpuValue: number | null;
    maxGpuValue: number | null;
  }>;
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    intermediateSize: number;
    heads: number;
    headDim: number;
    epsilon: number;
    tolerance: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4VisionImageFeaturesKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  layerCount: number;
  comparisonMode: "propagated" | "gpuAnchored";
  gpuPath: {
    completed: boolean;
    finite: boolean;
    outputRows: number;
    outputDim: number;
    checksum: number | null;
    firstValues: number[];
  };
  source: { width: number; height: number };
  layout: Gemma4ImageTokenLayout | null;
  gpu: {
    adapter: boolean;
    device: boolean;
    patches: number;
    softTokens: number;
    hiddenSize: number;
    textHiddenSize: number;
    poolingKernelSize: number;
    tolerance: number;
    workgroupSize: number;
  };
  comparisons: {
    patchEmbeddingMaxAbsDiff: number | null;
    encoderStackMaxAbsDiff: number | null;
    poolerMaxAbsDiff: number | null;
    embedderMaxAbsDiff: number | null;
    finalCpuChecksum: number | null;
    finalGpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  layers: Gemma4VisionEncoderStackKernelProbe["layers"];
  error?: string;
}

export type Gemma4VisionImageGpuPathKernelProbe = Gemma4VisionImageFeaturesKernelProbe & {
  strictComparisonOk: boolean;
};

export interface Gemma4VisionImageEmbeddingResult {
  ok: boolean;
  modelId: string;
  durationMs: number;
  source: { width: number; height: number };
  layout: Gemma4ImageTokenLayout | null;
  layerCount: number;
  rows: number;
  dim: number;
  embeddings: Float32Array;
  finite: boolean;
  checksum: number | null;
  firstValues: number[];
  error?: string;
}

export interface Gemma4AudioEmbeddingResult {
  ok: boolean;
  modelId: string;
  durationMs: number;
  source: { samples: number; samplingRate: number };
  layerCount: number;
  rows: number;
  dim: number;
  embeddings: Float32Array;
  finite: boolean;
  checksum: number | null;
  firstValues: number[];
  stages: {
    frames: number;
    validFrames: number;
    subsampleRows: number;
    validSubsampleRows: number;
    stackFinite: boolean;
    outputProjectionFinite: boolean;
  };
  error?: string;
}

export type Gemma4AudioEmbeddingKernelProbe = Omit<Gemma4AudioEmbeddingResult, "embeddings">;

export interface Gemma4MediaEmbedderKernelProbe {
  ok: boolean;
  modelId: string;
  kind: "vision" | "audio";
  durationMs: number;
  tensor: {
    name: string;
    dtype: string;
    shape: number[];
    dataBytes: number;
    fromCache: boolean;
    checksum: number;
  } | null;
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    inputDim: number;
    outputDim: number;
    epsilon: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export type Gemma4VisionEmbedderKernelProbe = Gemma4MediaEmbedderKernelProbe;

export type Gemma4AudioEmbedderKernelProbe = Gemma4MediaEmbedderKernelProbe;

export interface Gemma4AudioSubsampleKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  source: { samples: number; samplingRate: number };
  tensors: {
    layer0Conv: Gemma4TensorSummary | null;
    layer0Norm: Gemma4TensorSummary | null;
    layer1Conv: Gemma4TensorSummary | null;
    layer1Norm: Gemma4TensorSummary | null;
    inputProjection: Gemma4TensorSummary | null;
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    frames: number;
    featureSize: number;
    outputRows: number;
    validRows: number;
    hiddenSize: number;
    tolerance: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4AudioFeedForwardProjectionTensorSummary {
  projection: Gemma4AudioFeedForwardProjectionName;
  bits: number;
  weight: Gemma4TensorSummary | null;
  weightScale: Gemma4TensorSummary | null;
  inputActivationScale: number | null;
  outputActivationScale: number | null;
}

export interface Gemma4AudioFeedForwardKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  layerIndex: number;
  block: Gemma4AudioFeedForwardBlockName;
  tensors: {
    preLayerNorm: Gemma4TensorSummary | null;
    postLayerNorm: Gemma4TensorSummary | null;
    projections: Gemma4AudioFeedForwardProjectionTensorSummary[];
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    intermediateSize: number;
    activation: "silu";
    residualWeight: number;
    epsilon: number;
    tolerance: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4AudioLightConvProjectionTensorSummary {
  projection: Gemma4AudioLightConvProjectionName;
  bits: number;
  weight: Gemma4TensorSummary | null;
  weightScale: Gemma4TensorSummary | null;
  inputActivationScale: number | null;
  outputActivationScale: number | null;
}

export interface Gemma4AudioLightConvKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  layerIndex: number;
  tensors: {
    preLayerNorm: Gemma4TensorSummary | null;
    convNorm: Gemma4TensorSummary | null;
    depthwiseConv: Gemma4TensorSummary | null;
    projections: Gemma4AudioLightConvProjectionTensorSummary[];
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    expandedSize: number;
    kernelSize: number;
    activation: "glu+silu";
    epsilon: number;
    tolerance: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4AudioSelfAttentionProjectionTensorSummary {
  projection: Gemma4AudioSelfAttentionProjectionName;
  bits: number;
  weight: Gemma4TensorSummary | null;
  weightScale: Gemma4TensorSummary | null;
  inputActivationScale: number | null;
  outputActivationScale: number | null;
}

export interface Gemma4AudioSelfAttentionKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  layerIndex: number;
  tensors: {
    perDimScale: Gemma4TensorSummary | null;
    relativeKeyProjection: Gemma4TensorSummary | null;
    projections: Gemma4AudioSelfAttentionProjectionTensorSummary[];
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    heads: number;
    headDim: number;
    chunkSize: number;
    contextSize: number;
    positionLength: number;
    softcap: number;
    tolerance: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4AudioEncoderLayerKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  layerIndex: number;
  tensors: {
    normPreAttention: Gemma4TensorSummary | null;
    normPostAttention: Gemma4TensorSummary | null;
    normOut: Gemma4TensorSummary | null;
    feedForward1: Gemma4AudioFeedForwardProjectionTensorSummary[];
    selfAttention: Gemma4AudioSelfAttentionProjectionTensorSummary[];
    lightConv: Gemma4AudioLightConvProjectionTensorSummary[];
    feedForward2: Gemma4AudioFeedForwardProjectionTensorSummary[];
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    intermediateSize: number;
    expandedSize: number;
    heads: number;
    headDim: number;
    chunkSize: number;
    contextSize: number;
    positionLength: number;
    kernelSize: number;
    residualWeight: number;
    epsilon: number;
    tolerance: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    stageMaxAbsDiffs: Record<string, number>;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4AudioEncoderStackKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  layerCount: number;
  layers: Array<{
    layerIndex: number;
    maxAbsDiff: number;
    stageMaxAbsDiffs: Record<string, number>;
    cpuChecksum: number;
    gpuChecksum: number;
    firstCpuValues: number[];
    firstGpuValues: number[];
  }>;
  gpuPath: {
    completed: boolean;
    finite: boolean;
    checksum: number | null;
    firstValues: number[];
  };
  gpu: {
    adapter: boolean;
    device: boolean;
    rows: number;
    hiddenSize: number;
    intermediateSize: number;
    expandedSize: number;
    heads: number;
    headDim: number;
    chunkSize: number;
    contextSize: number;
    positionLength: number;
    kernelSize: number;
    residualWeight: number;
    epsilon: number;
    tolerance: number;
    workgroupSize: number;
  };
  comparison: {
    maxAbsDiff: number | null;
    cpuChecksum: number | null;
    gpuChecksum: number | null;
    firstCpuValues: number[];
    firstGpuValues: number[];
  };
  error?: string;
}

export interface Gemma4AudioEncoderStackStepKernelProbe {
  ok: boolean;
  modelId: string;
  durationMs: number;
  sessionId: number | null;
  completed: boolean;
  processedLayers: number;
  nextLayerIndex: number;
  totalLayers: number;
  layers: Gemma4AudioEncoderStackKernelProbe["layers"];
  gpuPath: Gemma4AudioEncoderStackKernelProbe["gpuPath"];
  gpu: Gemma4AudioEncoderStackKernelProbe["gpu"];
  comparison: Gemma4AudioEncoderStackKernelProbe["comparison"];
  error?: string;
}

export interface Gemma4TensorSummary {
  name: string;
  dtype: string;
  shape: number[];
  dataBytes: number;
  fromCache: boolean;
  checksum: number;
}

export interface Gemma4ImagePreprocessOptions {
  patchSize?: number;
  maxSoftTokens?: number;
  poolingKernelSize?: number;
  rescaleFactor?: number;
}

export interface Gemma4ImagePreprocessResult {
  layout: Gemma4ImageTokenLayout;
  pixelValues: Float32Array;
  imagePositionIds: Int32Array;
  numSoftTokens: number;
  maxPatches: number;
  patchPixels: number;
}

export interface Gemma4ImagePreprocessProbe {
  ok: boolean;
  source: { height: number; width: number };
  layout: Gemma4ImageTokenLayout;
  pixelValueLength: number;
  imagePositionIdLength: number;
  numSoftTokens: number;
  maxPatches: number;
  patchPixels: number;
  firstPositionIds: number[];
  paddedPositionTail: number[];
  pixelChecksum: number;
  error?: string;
}

export interface Gemma4AudioPreprocessOptions {
  samplingRate?: number;
  featureSize?: number;
  frameLength?: number;
  hopLength?: number;
  fftLength?: number;
  minFrequency?: number;
  maxFrequency?: number;
  melFloor?: number;
  inputScaleFactor?: number;
  padToMultipleOf?: number | null;
  maxSamples?: number;
  audioSeqLength?: number;
}

export interface Gemma4AudioPreprocessResult {
  inputFeatures: Float32Array;
  inputFeaturesMask: Uint8Array;
  frames: number;
  validFrames: number;
  featureSize: number;
  samplingRate: number;
  fftLength: number;
  softTokens: number;
}

export interface Gemma4AudioPreprocessProbe {
  ok: boolean;
  source: { samples: number; samplingRate: number };
  frames: number;
  validFrames: number;
  featureSize: number;
  inputFeatureLength: number;
  inputFeatureMaskLength: number;
  softTokens: number;
  firstMaskValues: number[];
  featureChecksum: number;
  error?: string;
}

export interface AudioEncoderStackStepSession {
  id: number;
  cpuHiddenStates: Float32Array;
  gpuHiddenStates: Float32Array;
  nextLayerIndex: number;
  layers: Gemma4AudioEncoderStackKernelProbe["layers"];
}

export interface AudioSubsampleWeights {
  layer0Conv: Float32Array;
  layer0Norm: Float32Array;
  layer1Conv: Float32Array;
  layer1Norm: Float32Array;
  inputProjection: Float32Array;
}

export interface AudioSubsampleProjectionInput {
  inputFeatures: Float32Array;
  inputFeaturesMask: Uint8Array;
  frames: number;
  featureSize: number;
  weights: AudioSubsampleWeights;
  hiddenSize: number;
  epsilon: number;
}

export interface AudioSubsampleProjectionOutput {
  output: Float32Array;
  mask: Uint8Array;
  rows: number;
  validRows: number;
}

export interface AudioSubsampleConvLayerInput {
  inputValues: Float32Array;
  inputMask: Uint8Array;
  inputTime: number;
  inputFreq: number;
  inputChannels: number;
  weights: Float32Array;
  normWeights: Float32Array;
  outputChannels: number;
  epsilon: number;
}

export interface AudioSubsampleConvLayerOutput {
  output: Float32Array;
  mask: Uint8Array;
  outputTime: number;
  outputFreq: number;
}

export interface PackedQatProjectionData<TProjection extends string = string> {
  projection: TProjection;
  weightData: SafetensorsTensorData;
  weightScaleData: SafetensorsTensorData;
  bits: number;
  inputActivationScale: number;
  outputActivationScale: number;
}

export interface PackedQatLinearCpuInput {
  input: Float32Array;
  weights: Uint8Array;
  weightScales: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  bits: number;
  inputActivationScale: number;
  outputActivationScale: number;
}

export interface PackedQatLinearGpuInput {
  input: Float32Array;
  weights: Uint8Array;
  weightScales: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  bits: number;
  inputActivationScale: number;
  outputActivationScale: number;
  label: string;
}

export interface AudioSelfAttentionBodyInput {
  query: Float32Array;
  key: Float32Array;
  value: Float32Array;
  relativeKey: Float32Array;
  perDimScale: Float32Array;
  rows: number;
  hiddenSize: number;
  heads: number;
  headDim: number;
  chunkSize: number;
  contextSize: number;
  positionLength: number;
  softcap: number;
}

export interface QatI8LinearInput {
  input: Float32Array;
  weights: Int8Array;
  weightScales: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  inputActivationScale: number;
  outputActivationScale: number;
}

export interface VisionAttentionBodyInput {
  query: Float32Array;
  key: Float32Array;
  value: Float32Array;
  rows: number;
  heads: number;
  headDim: number;
  scaling: number;
}

export interface MultimodalEmbedderGpuInput {
  hiddenStates: Float32Array;
  weights: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  epsilon: number;
  label: string;
}

export interface QatI8LinearGpuInput {
  input: Float32Array;
  weights: Uint8Array;
  weightScales: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  inputActivationScale: number;
  outputActivationScale: number;
  label: string;
}

export interface VisionPoolerInput {
  hiddenStates: Float32Array;
  positions: Array<{ x: number; y: number }>;
  outputLength: number;
  hiddenSize: number;
  poolingKernelSize: number;
}

export interface VisionPatchEmbeddingInput {
  patchValues: Float32Array;
  projectionWeights: Float32Array;
  positions: Array<{ x: number; y: number }>;
  positionXRows: Float32Array;
  positionYRows: Float32Array;
  patchPixels: number;
  hiddenSize: number;
  outputRows: number;
}

export type NavigatorWithGpu = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<{
      requestDevice: () => Promise<GPUDevice>;
    } | null>;
  };
};