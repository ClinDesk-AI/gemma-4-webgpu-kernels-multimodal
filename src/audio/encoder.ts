import { audioRelativePositionEmbeddings, countMask, runAudioDepthwiseCausalConv1dCpu, runAudioDepthwiseCausalConv1dWebGpu, runAudioSelfAttentionBodyCpu, runAudioSelfAttentionBodyWebGpu, runGluActivationCpu, runGluActivationWebGpu, runScaledResidualAddCpu, runScaledResidualAddWebGpu, runSiluActivationCpu, runSiluActivationWebGpu } from "./kernels.js";
import { assertF32Tensor, float32FromBytes, loadSafetensorsTensorData } from "../io/safetensors.js";
import { loadPackedQatProjection, runF32LinearRowsCpu, runF32LinearRowsWebGpu, runPackedQatLinearCpu, runPackedQatLinearWebGpu } from "../kernels/qat.js";
import { audioEncoderLayerNormTensors, audioFeedForwardNormTensors, audioFeedForwardProjectionTensors, audioLightConvProjectionTensors, audioLightConvTensors, audioSelfAttentionProjectionTensors, audioSelfAttentionTensors } from "../model.js";
import { Gemma4AudioEncoderStackKernelProbe, Gemma4AudioFeedForwardBlockName, Gemma4AudioFeedForwardProjectionName, Gemma4AudioFeedForwardProjectionTensorSummary, Gemma4AudioLightConvProjectionName, Gemma4AudioLightConvProjectionTensorSummary, Gemma4AudioSelfAttentionProjectionName, Gemma4AudioSelfAttentionProjectionTensorSummary, Gemma4MediaKernelProbe, PackedQatProjectionData, SafetensorsHeader } from "../types.js";
import { checksumFloats, maxAbsDifference, roundedSample, tensorSummaryForProbe } from "../utils/math.js";
import { runResidualAddCpu, runResidualAddWebGpu, runVisionRmsNormCpu, runVisionRmsNormWebGpu } from "../vision/kernels.js";

export async function computeAudioFeedForward(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  block: Gemma4AudioFeedForwardBlockName;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  epsilon: number;
  residualWeight: number;
  signal?: AbortSignal;
}): Promise<{
  projections: Array<PackedQatProjectionData<Gemma4AudioFeedForwardProjectionName>>;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
}> {
  const normNames = audioFeedForwardNormTensors(input.layerIndex, input.block);
  const [preLayerNormData, postLayerNormData, layer1, layer2] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.preLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.postLayerNorm, input.signal),
    loadAudioFeedForwardProjection(input, "ffw_layer_1", input.hiddenSize, input.intermediateSize),
    loadAudioFeedForwardProjection(input, "ffw_layer_2", input.intermediateSize, input.hiddenSize),
  ]);
  assertF32Tensor(
    preLayerNormData,
    [input.hiddenSize],
    `audio layer ${input.layerIndex} ${input.block} pre layernorm`,
  );
  assertF32Tensor(
    postLayerNormData,
    [input.hiddenSize],
    `audio layer ${input.layerIndex} ${input.block} post layernorm`,
  );

  const preLayerNormWeights = float32FromBytes(preLayerNormData.bytes);
  const postLayerNormWeights = float32FromBytes(postLayerNormData.bytes);
  const gpuHiddenStates = input.gpuHiddenStates ?? input.hiddenStates;
  const normalizedCpu = runVisionRmsNormCpu(
    input.hiddenStates,
    preLayerNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const normalizedGpu = await runVisionRmsNormWebGpu(
    input.device,
    gpuHiddenStates,
    preLayerNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const layer1Cpu = runPackedQatLinearCpu({
    input: normalizedCpu,
    weights: layer1.weightData.bytes,
    weightScales: float32FromBytes(layer1.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.intermediateSize,
    bits: layer1.bits,
    inputActivationScale: layer1.inputActivationScale,
    outputActivationScale: layer1.outputActivationScale,
  });
  const layer1Gpu = await runPackedQatLinearWebGpu(input.device, {
    input: normalizedGpu,
    weights: layer1.weightData.bytes,
    weightScales: float32FromBytes(layer1.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.intermediateSize,
    bits: layer1.bits,
    inputActivationScale: layer1.inputActivationScale,
    outputActivationScale: layer1.outputActivationScale,
    label: `gemma4-audio-${input.block}-ffw-layer-1`,
  });
  const activatedCpu = runSiluActivationCpu(layer1Cpu);
  const activatedGpu = await runSiluActivationWebGpu(input.device, layer1Gpu, `gemma4-audio-${input.block}-silu`);
  const layer2WeightScales = float32FromBytes(layer2.weightScaleData.bytes);
  const layer2Cpu = runPackedQatLinearCpu({
    input: activatedCpu,
    weights: layer2.weightData.bytes,
    weightScales: layer2WeightScales,
    rows: input.rows,
    inputDim: input.intermediateSize,
    outputDim: input.hiddenSize,
    bits: layer2.bits,
    inputActivationScale: layer2.inputActivationScale,
    outputActivationScale: layer2.outputActivationScale,
  });
  const layer2Gpu = await runPackedQatLinearWebGpu(input.device, {
    input: activatedGpu,
    weights: layer2.weightData.bytes,
    weightScales: layer2WeightScales,
    rows: input.rows,
    inputDim: input.intermediateSize,
    outputDim: input.hiddenSize,
    bits: layer2.bits,
    inputActivationScale: layer2.inputActivationScale,
    outputActivationScale: layer2.outputActivationScale,
    label: `gemma4-audio-${input.block}-ffw-layer-2`,
  });
  const postNormCpu = runVisionRmsNormCpu(
    layer2Cpu,
    postLayerNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const postNormGpu = await runVisionRmsNormWebGpu(
    input.device,
    layer2Gpu,
    postLayerNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );

  return {
    projections: [layer1, layer2],
    cpuOutput: runScaledResidualAddCpu(input.hiddenStates, postNormCpu, input.residualWeight),
    gpuOutput: await runScaledResidualAddWebGpu(
      input.device,
      gpuHiddenStates,
      postNormGpu,
      input.residualWeight,
      `gemma4-audio-${input.block}-residual`,
    ),
  };
}

export async function loadAudioFeedForwardProjection(
  input: {
    urls: Gemma4MediaKernelProbe["urls"];
    header: SafetensorsHeader;
    layerIndex: number;
    block: Gemma4AudioFeedForwardBlockName;
    signal?: AbortSignal;
  },
  projection: Gemma4AudioFeedForwardProjectionName,
  inputDim: number,
  outputDim: number,
): Promise<PackedQatProjectionData<Gemma4AudioFeedForwardProjectionName>> {
  const names = audioFeedForwardProjectionTensors(input.layerIndex, input.block, projection);
  return loadPackedQatProjection({
    urls: input.urls,
    header: input.header,
    names,
    projection,
    inputDim,
    outputDim,
    signal: input.signal,
  });
}

export async function loadAudioLightConvProjection(
  input: {
    urls: Gemma4MediaKernelProbe["urls"];
    header: SafetensorsHeader;
    layerIndex: number;
    signal?: AbortSignal;
  },
  projection: Gemma4AudioLightConvProjectionName,
  inputDim: number,
  outputDim: number,
): Promise<PackedQatProjectionData<Gemma4AudioLightConvProjectionName>> {
  return loadPackedQatProjection({
    urls: input.urls,
    header: input.header,
    names: audioLightConvProjectionTensors(input.layerIndex, projection),
    projection,
    inputDim,
    outputDim,
    signal: input.signal,
  });
}

export function audioFeedForwardProjectionSummary(
  input: PackedQatProjectionData<Gemma4AudioFeedForwardProjectionName>,
): Gemma4AudioFeedForwardProjectionTensorSummary {
  return {
    projection: input.projection,
    bits: input.bits,
    weight: tensorSummaryForProbe(input.weightData),
    weightScale: tensorSummaryForProbe(input.weightScaleData),
    inputActivationScale: input.inputActivationScale,
    outputActivationScale: input.outputActivationScale,
  };
}

export function audioLightConvProjectionSummary(
  input: PackedQatProjectionData<Gemma4AudioLightConvProjectionName>,
): Gemma4AudioLightConvProjectionTensorSummary {
  return {
    projection: input.projection,
    bits: input.bits,
    weight: tensorSummaryForProbe(input.weightData),
    weightScale: tensorSummaryForProbe(input.weightScaleData),
    inputActivationScale: input.inputActivationScale,
    outputActivationScale: input.outputActivationScale,
  };
}

export function audioSelfAttentionProjectionSummary(
  input: PackedQatProjectionData<Gemma4AudioSelfAttentionProjectionName>,
): Gemma4AudioSelfAttentionProjectionTensorSummary {
  return {
    projection: input.projection,
    bits: input.bits,
    weight: tensorSummaryForProbe(input.weightData),
    weightScale: tensorSummaryForProbe(input.weightScaleData),
    inputActivationScale: input.inputActivationScale,
    outputActivationScale: input.outputActivationScale,
  };
}

export async function computeAudioLightConv(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  rows: number;
  hiddenSize: number;
  expandedSize: number;
  kernelSize: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<{
  projections: Array<PackedQatProjectionData<Gemma4AudioLightConvProjectionName>>;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
}> {
  const tensorNames = audioLightConvTensors(input.layerIndex);
  const [preLayerNormData, convNormData, depthwiseConvData, linearStart, linearEnd] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.preLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.convNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.depthwiseConv, input.signal),
    loadAudioLightConvProjection(input, "linear_start", input.hiddenSize, input.expandedSize),
    loadAudioLightConvProjection(input, "linear_end", input.hiddenSize, input.hiddenSize),
  ]);
  assertF32Tensor(preLayerNormData, [input.hiddenSize], `audio layer ${input.layerIndex} lconv pre layernorm`);
  assertF32Tensor(convNormData, [input.hiddenSize], `audio layer ${input.layerIndex} lconv norm`);
  assertF32Tensor(
    depthwiseConvData,
    [input.hiddenSize, 1, input.kernelSize],
    `audio layer ${input.layerIndex} lconv depthwise conv`,
  );

  const preLayerNormWeights = float32FromBytes(preLayerNormData.bytes);
  const convNormWeights = float32FromBytes(convNormData.bytes);
  const depthwiseWeights = float32FromBytes(depthwiseConvData.bytes);
  const gpuHiddenStates = input.gpuHiddenStates ?? input.hiddenStates;
  const normalizedCpu = runVisionRmsNormCpu(
    input.hiddenStates,
    preLayerNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const normalizedGpu = await runVisionRmsNormWebGpu(
    input.device,
    gpuHiddenStates,
    preLayerNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const startWeightScales = float32FromBytes(linearStart.weightScaleData.bytes);
  const linearStartCpu = runPackedQatLinearCpu({
    input: normalizedCpu,
    weights: linearStart.weightData.bytes,
    weightScales: startWeightScales,
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.expandedSize,
    bits: linearStart.bits,
    inputActivationScale: linearStart.inputActivationScale,
    outputActivationScale: linearStart.outputActivationScale,
  });
  const linearStartGpu = await runPackedQatLinearWebGpu(input.device, {
    input: normalizedGpu,
    weights: linearStart.weightData.bytes,
    weightScales: startWeightScales,
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.expandedSize,
    bits: linearStart.bits,
    inputActivationScale: linearStart.inputActivationScale,
    outputActivationScale: linearStart.outputActivationScale,
    label: "gemma4-audio-lconv-linear-start",
  });
  const gluCpu = runGluActivationCpu(linearStartCpu, input.rows, input.hiddenSize);
  const gluGpu = await runGluActivationWebGpu(
    input.device,
    linearStartGpu,
    input.rows,
    input.hiddenSize,
    "gemma4-audio-lconv-glu",
  );
  const convCpu = runAudioDepthwiseCausalConv1dCpu({
    input: gluCpu,
    weights: depthwiseWeights,
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    kernelSize: input.kernelSize,
  });
  const convGpu = await runAudioDepthwiseCausalConv1dWebGpu(input.device, {
    input: gluGpu,
    weights: depthwiseWeights,
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    kernelSize: input.kernelSize,
    label: "gemma4-audio-lconv-depthwise",
  });
  const convNormCpu = runVisionRmsNormCpu(
    convCpu,
    convNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const convNormGpu = await runVisionRmsNormWebGpu(
    input.device,
    convGpu,
    convNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const activatedCpu = runSiluActivationCpu(convNormCpu);
  const activatedGpu = await runSiluActivationWebGpu(input.device, convNormGpu, "gemma4-audio-lconv-silu");
  const endWeightScales = float32FromBytes(linearEnd.weightScaleData.bytes);
  const linearEndCpu = runPackedQatLinearCpu({
    input: activatedCpu,
    weights: linearEnd.weightData.bytes,
    weightScales: endWeightScales,
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    bits: linearEnd.bits,
    inputActivationScale: linearEnd.inputActivationScale,
    outputActivationScale: linearEnd.outputActivationScale,
  });
  const linearEndGpu = await runPackedQatLinearWebGpu(input.device, {
    input: activatedGpu,
    weights: linearEnd.weightData.bytes,
    weightScales: endWeightScales,
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    bits: linearEnd.bits,
    inputActivationScale: linearEnd.inputActivationScale,
    outputActivationScale: linearEnd.outputActivationScale,
    label: "gemma4-audio-lconv-linear-end",
  });

  return {
    projections: [linearStart, linearEnd],
    cpuOutput: runResidualAddCpu(input.hiddenStates, linearEndCpu),
    gpuOutput: await runResidualAddWebGpu(input.device, gpuHiddenStates, linearEndGpu, "gemma4-audio-lconv-residual"),
  };
}

export async function computeAudioSelfAttention(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  rows: number;
  hiddenSize: number;
  heads: number;
  headDim: number;
  chunkSize: number;
  contextSize: number;
  positionLength: number;
  softcap: number;
  signal?: AbortSignal;
}): Promise<{
  projections: Array<PackedQatProjectionData<Gemma4AudioSelfAttentionProjectionName>>;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
}> {
  const tensorNames = audioSelfAttentionTensors(input.layerIndex);
  const [perDimScaleData, relativeKeyProjectionData, qProj, kProj, vProj, postProj] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.perDimScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.relativeKeyProjection, input.signal),
    loadAudioSelfAttentionProjection(input, "q_proj", input.hiddenSize, input.hiddenSize),
    loadAudioSelfAttentionProjection(input, "k_proj", input.hiddenSize, input.hiddenSize),
    loadAudioSelfAttentionProjection(input, "v_proj", input.hiddenSize, input.hiddenSize),
    loadAudioSelfAttentionProjection(input, "post", input.hiddenSize, input.hiddenSize),
  ]);
  assertF32Tensor(perDimScaleData, [input.headDim], `audio layer ${input.layerIndex} attention per-dim scale`);
  assertF32Tensor(
    relativeKeyProjectionData,
    [input.hiddenSize, input.hiddenSize],
    `audio layer ${input.layerIndex} relative key projection`,
  );

  const gpuHiddenStates = input.gpuHiddenStates ?? input.hiddenStates;
  const q = await runPackedProjectionPair(input, qProj, input.hiddenStates, "gemma4-audio-attn-q-proj", gpuHiddenStates);
  const k = await runPackedProjectionPair(input, kProj, input.hiddenStates, "gemma4-audio-attn-k-proj", gpuHiddenStates);
  const v = await runPackedProjectionPair(input, vProj, input.hiddenStates, "gemma4-audio-attn-v-proj", gpuHiddenStates);
  const positionEmbeddings = audioRelativePositionEmbeddings(input.positionLength, input.hiddenSize);
  const relativeWeights = float32FromBytes(relativeKeyProjectionData.bytes);
  const relativeKeysCpu = runF32LinearRowsCpu(
    positionEmbeddings,
    relativeWeights,
    input.positionLength,
    input.hiddenSize,
    input.hiddenSize,
  );
  const relativeKeysGpu = await runF32LinearRowsWebGpu(input.device, {
    input: positionEmbeddings,
    weights: relativeWeights,
    rows: input.positionLength,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    label: "gemma4-audio-attn-relative-k-proj",
  });
  const bodyCpu = runAudioSelfAttentionBodyCpu({
    query: q.cpuOutput,
    key: k.cpuOutput,
    value: v.cpuOutput,
    relativeKey: relativeKeysCpu,
    perDimScale: float32FromBytes(perDimScaleData.bytes),
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    heads: input.heads,
    headDim: input.headDim,
    chunkSize: input.chunkSize,
    contextSize: input.contextSize,
    positionLength: input.positionLength,
    softcap: input.softcap,
  });
  const bodyGpu = await runAudioSelfAttentionBodyWebGpu(input.device, {
    query: q.gpuOutput,
    key: k.gpuOutput,
    value: v.gpuOutput,
    relativeKey: relativeKeysGpu,
    perDimScale: float32FromBytes(perDimScaleData.bytes),
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    heads: input.heads,
    headDim: input.headDim,
    chunkSize: input.chunkSize,
    contextSize: input.contextSize,
    positionLength: input.positionLength,
    softcap: input.softcap,
    label: "gemma4-audio-attn-body",
  });
  const post = await runPackedProjectionPair(input, postProj, bodyCpu, "gemma4-audio-attn-post-proj", bodyGpu);

  return {
    projections: [qProj, kProj, vProj, postProj],
    cpuOutput: post.cpuOutput,
    gpuOutput: post.gpuOutput,
  };
}

export async function computeAudioEncoderLayer(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
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
  softcap: number;
  residualWeight: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<{
  feedForward1: Awaited<ReturnType<typeof computeAudioFeedForward>>;
  selfAttention: Awaited<ReturnType<typeof computeAudioSelfAttention>>;
  lightConv: Awaited<ReturnType<typeof computeAudioLightConv>>;
  feedForward2: Awaited<ReturnType<typeof computeAudioFeedForward>>;
  stageMaxAbsDiffs: Record<string, number>;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
}> {
  const tensorNames = audioEncoderLayerNormTensors(input.layerIndex);
  const [normPreAttentionData, normPostAttentionData, normOutData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.normPreAttention, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.normPostAttention, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.normOut, input.signal),
  ]);
  assertF32Tensor(normPreAttentionData, [input.hiddenSize], `audio layer ${input.layerIndex} pre-attention norm`);
  assertF32Tensor(normPostAttentionData, [input.hiddenSize], `audio layer ${input.layerIndex} post-attention norm`);
  assertF32Tensor(normOutData, [input.hiddenSize], `audio layer ${input.layerIndex} output norm`);

  const stageMaxAbsDiffs: Record<string, number> = {};
  const feedForward1 = await computeAudioFeedForward({
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    block: "feed_forward1",
    hiddenStates: input.hiddenStates,
    gpuHiddenStates: input.gpuHiddenStates,
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    intermediateSize: input.intermediateSize,
    epsilon: input.epsilon,
    residualWeight: input.residualWeight,
    signal: input.signal,
  });
  stageMaxAbsDiffs.afterFeedForward1 = maxAbsDifference(feedForward1.cpuOutput, feedForward1.gpuOutput);

  const preAttentionWeights = float32FromBytes(normPreAttentionData.bytes);
  const attentionInputCpu = runVisionRmsNormCpu(
    feedForward1.cpuOutput,
    preAttentionWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const attentionInputGpu = await runVisionRmsNormWebGpu(
    input.device,
    feedForward1.gpuOutput,
    preAttentionWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  stageMaxAbsDiffs.afterPreAttentionNorm = maxAbsDifference(attentionInputCpu, attentionInputGpu);

  const selfAttention = await computeAudioSelfAttention({
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    hiddenStates: attentionInputCpu,
    gpuHiddenStates: attentionInputGpu,
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    heads: input.heads,
    headDim: input.headDim,
    chunkSize: input.chunkSize,
    contextSize: input.contextSize,
    positionLength: input.positionLength,
    softcap: input.softcap,
    signal: input.signal,
  });
  stageMaxAbsDiffs.afterSelfAttention = maxAbsDifference(selfAttention.cpuOutput, selfAttention.gpuOutput);

  const postAttentionWeights = float32FromBytes(normPostAttentionData.bytes);
  const postAttentionNormCpu = runVisionRmsNormCpu(
    selfAttention.cpuOutput,
    postAttentionWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const postAttentionNormGpu = await runVisionRmsNormWebGpu(
    input.device,
    selfAttention.gpuOutput,
    postAttentionWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const attentionResidualCpu = runResidualAddCpu(feedForward1.cpuOutput, postAttentionNormCpu);
  const attentionResidualGpu = await runResidualAddWebGpu(
    input.device,
    feedForward1.gpuOutput,
    postAttentionNormGpu,
    "gemma4-audio-layer-attn-residual",
  );
  stageMaxAbsDiffs.afterAttentionResidual = maxAbsDifference(attentionResidualCpu, attentionResidualGpu);

  const lightConv = await computeAudioLightConv({
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    hiddenStates: attentionResidualCpu,
    gpuHiddenStates: attentionResidualGpu,
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    expandedSize: input.expandedSize,
    kernelSize: input.kernelSize,
    epsilon: input.epsilon,
    signal: input.signal,
  });
  stageMaxAbsDiffs.afterLightConv = maxAbsDifference(lightConv.cpuOutput, lightConv.gpuOutput);

  const feedForward2 = await computeAudioFeedForward({
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    block: "feed_forward2",
    hiddenStates: lightConv.cpuOutput,
    gpuHiddenStates: lightConv.gpuOutput,
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    intermediateSize: input.intermediateSize,
    epsilon: input.epsilon,
    residualWeight: input.residualWeight,
    signal: input.signal,
  });
  stageMaxAbsDiffs.afterFeedForward2 = maxAbsDifference(feedForward2.cpuOutput, feedForward2.gpuOutput);

  const normOutWeights = float32FromBytes(normOutData.bytes);
  const cpuOutput = runVisionRmsNormCpu(
    feedForward2.cpuOutput,
    normOutWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const gpuOutput = await runVisionRmsNormWebGpu(
    input.device,
    feedForward2.gpuOutput,
    normOutWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  stageMaxAbsDiffs.final = maxAbsDifference(cpuOutput, gpuOutput);

  return {
    feedForward1,
    selfAttention,
    lightConv,
    feedForward2,
    stageMaxAbsDiffs,
    cpuOutput,
    gpuOutput,
  };
}

export async function computeAudioEncoderStack(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerCount: number;
  hiddenStates: Float32Array;
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
  softcap: number;
  residualWeight: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<{
  layers: Gemma4AudioEncoderStackKernelProbe["layers"];
  maxAbsDiff: number;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
}> {
  let cpuHiddenStates = input.hiddenStates;
  let gpuHiddenStates = input.hiddenStates;
  const layers: Gemma4AudioEncoderStackKernelProbe["layers"] = [];
  for (let layerIndex = 0; layerIndex < input.layerCount; layerIndex += 1) {
    const layer = await computeAudioEncoderLayer({
      device: input.device,
      urls: input.urls,
      header: input.header,
      layerIndex,
      hiddenStates: cpuHiddenStates,
      gpuHiddenStates,
      rows: input.rows,
      hiddenSize: input.hiddenSize,
      intermediateSize: input.intermediateSize,
      expandedSize: input.expandedSize,
      heads: input.heads,
      headDim: input.headDim,
      chunkSize: input.chunkSize,
      contextSize: input.contextSize,
      positionLength: input.positionLength,
      kernelSize: input.kernelSize,
      softcap: input.softcap,
      residualWeight: input.residualWeight,
      epsilon: input.epsilon,
      signal: input.signal,
    });
    cpuHiddenStates = layer.cpuOutput;
    gpuHiddenStates = layer.gpuOutput;
    const maxAbsDiff = maxAbsDifference(cpuHiddenStates, gpuHiddenStates);
    layers.push({
      layerIndex,
      maxAbsDiff,
      stageMaxAbsDiffs: layer.stageMaxAbsDiffs,
      cpuChecksum: checksumFloats(cpuHiddenStates),
      gpuChecksum: checksumFloats(gpuHiddenStates),
      firstCpuValues: roundedSample(cpuHiddenStates),
      firstGpuValues: roundedSample(gpuHiddenStates),
    });
  }
  return {
    layers,
    maxAbsDiff: maxAbsDifference(cpuHiddenStates, gpuHiddenStates),
    cpuOutput: cpuHiddenStates,
    gpuOutput: gpuHiddenStates,
  };
}

export async function computeAudioEncoderStackGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerCount: number;
  hiddenStates: Float32Array;
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
  softcap: number;
  residualWeight: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  let gpuHiddenStates = input.hiddenStates;
  for (let layerIndex = 0; layerIndex < input.layerCount; layerIndex += 1) {
    gpuHiddenStates = await computeAudioEncoderLayerGpuOnly({
      ...input,
      layerIndex,
      hiddenStates: gpuHiddenStates,
    });
  }
  return gpuHiddenStates;
}

export async function computeAudioEncoderLayerGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  hiddenStates: Float32Array;
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
  softcap: number;
  residualWeight: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const tensorNames = audioEncoderLayerNormTensors(input.layerIndex);
  const [normPreAttentionData, normPostAttentionData, normOutData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.normPreAttention, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.normPostAttention, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.normOut, input.signal),
  ]);
  assertF32Tensor(normPreAttentionData, [input.hiddenSize], `audio layer ${input.layerIndex} pre-attention norm`);
  assertF32Tensor(normPostAttentionData, [input.hiddenSize], `audio layer ${input.layerIndex} post-attention norm`);
  assertF32Tensor(normOutData, [input.hiddenSize], `audio layer ${input.layerIndex} output norm`);

  const feedForward1 = await computeAudioFeedForwardGpuOnly({
    ...input,
    block: "feed_forward1",
  });
  const attentionInput = await runVisionRmsNormWebGpu(
    input.device,
    feedForward1,
    float32FromBytes(normPreAttentionData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const selfAttention = await computeAudioSelfAttentionGpuOnly({
    ...input,
    hiddenStates: attentionInput,
  });
  const postAttentionNorm = await runVisionRmsNormWebGpu(
    input.device,
    selfAttention,
    float32FromBytes(normPostAttentionData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const attentionResidual = await runResidualAddWebGpu(
    input.device,
    feedForward1,
    postAttentionNorm,
    "gemma4-audio-layer-attn-residual",
  );
  const lightConv = await computeAudioLightConvGpuOnly({
    ...input,
    hiddenStates: attentionResidual,
  });
  const feedForward2 = await computeAudioFeedForwardGpuOnly({
    ...input,
    hiddenStates: lightConv,
    block: "feed_forward2",
  });
  return runVisionRmsNormWebGpu(
    input.device,
    feedForward2,
    float32FromBytes(normOutData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
}

export async function computeAudioFeedForwardGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  block: Gemma4AudioFeedForwardBlockName;
  hiddenStates: Float32Array;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  epsilon: number;
  residualWeight: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const normNames = audioFeedForwardNormTensors(input.layerIndex, input.block);
  const [preLayerNormData, postLayerNormData, layer1, layer2] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.preLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.postLayerNorm, input.signal),
    loadAudioFeedForwardProjection(input, "ffw_layer_1", input.hiddenSize, input.intermediateSize),
    loadAudioFeedForwardProjection(input, "ffw_layer_2", input.intermediateSize, input.hiddenSize),
  ]);
  assertF32Tensor(
    preLayerNormData,
    [input.hiddenSize],
    `audio layer ${input.layerIndex} ${input.block} pre layernorm`,
  );
  assertF32Tensor(
    postLayerNormData,
    [input.hiddenSize],
    `audio layer ${input.layerIndex} ${input.block} post layernorm`,
  );

  const normalized = await runVisionRmsNormWebGpu(
    input.device,
    input.hiddenStates,
    float32FromBytes(preLayerNormData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const layer1Gpu = await runPackedQatLinearWebGpu(input.device, {
    input: normalized,
    weights: layer1.weightData.bytes,
    weightScales: float32FromBytes(layer1.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.intermediateSize,
    bits: layer1.bits,
    inputActivationScale: layer1.inputActivationScale,
    outputActivationScale: layer1.outputActivationScale,
    label: `gemma4-audio-${input.block}-ffw-layer-1`,
  });
  const activated = await runSiluActivationWebGpu(input.device, layer1Gpu, `gemma4-audio-${input.block}-silu`);
  const layer2Gpu = await runPackedQatLinearWebGpu(input.device, {
    input: activated,
    weights: layer2.weightData.bytes,
    weightScales: float32FromBytes(layer2.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.intermediateSize,
    outputDim: input.hiddenSize,
    bits: layer2.bits,
    inputActivationScale: layer2.inputActivationScale,
    outputActivationScale: layer2.outputActivationScale,
    label: `gemma4-audio-${input.block}-ffw-layer-2`,
  });
  const postNorm = await runVisionRmsNormWebGpu(
    input.device,
    layer2Gpu,
    float32FromBytes(postLayerNormData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  return runScaledResidualAddWebGpu(
    input.device,
    input.hiddenStates,
    postNorm,
    input.residualWeight,
    `gemma4-audio-${input.block}-residual`,
  );
}

export async function computeAudioLightConvGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  hiddenStates: Float32Array;
  rows: number;
  hiddenSize: number;
  expandedSize: number;
  kernelSize: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const tensorNames = audioLightConvTensors(input.layerIndex);
  const [preLayerNormData, convNormData, depthwiseConvData, linearStart, linearEnd] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.preLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.convNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.depthwiseConv, input.signal),
    loadAudioLightConvProjection(input, "linear_start", input.hiddenSize, input.expandedSize),
    loadAudioLightConvProjection(input, "linear_end", input.hiddenSize, input.hiddenSize),
  ]);
  assertF32Tensor(preLayerNormData, [input.hiddenSize], `audio layer ${input.layerIndex} lconv pre layernorm`);
  assertF32Tensor(convNormData, [input.hiddenSize], `audio layer ${input.layerIndex} lconv norm`);
  assertF32Tensor(
    depthwiseConvData,
    [input.hiddenSize, 1, input.kernelSize],
    `audio layer ${input.layerIndex} lconv depthwise conv`,
  );

  const normalized = await runVisionRmsNormWebGpu(
    input.device,
    input.hiddenStates,
    float32FromBytes(preLayerNormData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const linearStartGpu = await runPackedQatLinearWebGpu(input.device, {
    input: normalized,
    weights: linearStart.weightData.bytes,
    weightScales: float32FromBytes(linearStart.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.expandedSize,
    bits: linearStart.bits,
    inputActivationScale: linearStart.inputActivationScale,
    outputActivationScale: linearStart.outputActivationScale,
    label: "gemma4-audio-lconv-linear-start",
  });
  const glu = await runGluActivationWebGpu(
    input.device,
    linearStartGpu,
    input.rows,
    input.hiddenSize,
    "gemma4-audio-lconv-glu",
  );
  const conv = await runAudioDepthwiseCausalConv1dWebGpu(input.device, {
    input: glu,
    weights: float32FromBytes(depthwiseConvData.bytes),
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    kernelSize: input.kernelSize,
    label: "gemma4-audio-lconv-depthwise",
  });
  const convNorm = await runVisionRmsNormWebGpu(
    input.device,
    conv,
    float32FromBytes(convNormData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const activated = await runSiluActivationWebGpu(input.device, convNorm, "gemma4-audio-lconv-silu");
  const linearEndGpu = await runPackedQatLinearWebGpu(input.device, {
    input: activated,
    weights: linearEnd.weightData.bytes,
    weightScales: float32FromBytes(linearEnd.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    bits: linearEnd.bits,
    inputActivationScale: linearEnd.inputActivationScale,
    outputActivationScale: linearEnd.outputActivationScale,
    label: "gemma4-audio-lconv-linear-end",
  });
  return runResidualAddWebGpu(input.device, input.hiddenStates, linearEndGpu, "gemma4-audio-lconv-residual");
}

export async function computeAudioSelfAttentionGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  hiddenStates: Float32Array;
  rows: number;
  hiddenSize: number;
  heads: number;
  headDim: number;
  chunkSize: number;
  contextSize: number;
  positionLength: number;
  softcap: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const tensorNames = audioSelfAttentionTensors(input.layerIndex);
  const [perDimScaleData, relativeKeyProjectionData, qProj, kProj, vProj, postProj] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.perDimScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, tensorNames.relativeKeyProjection, input.signal),
    loadAudioSelfAttentionProjection(input, "q_proj", input.hiddenSize, input.hiddenSize),
    loadAudioSelfAttentionProjection(input, "k_proj", input.hiddenSize, input.hiddenSize),
    loadAudioSelfAttentionProjection(input, "v_proj", input.hiddenSize, input.hiddenSize),
    loadAudioSelfAttentionProjection(input, "post", input.hiddenSize, input.hiddenSize),
  ]);
  assertF32Tensor(perDimScaleData, [input.headDim], `audio layer ${input.layerIndex} attention per-dim scale`);
  assertF32Tensor(
    relativeKeyProjectionData,
    [input.hiddenSize, input.hiddenSize],
    `audio layer ${input.layerIndex} relative key projection`,
  );

  const q = await runPackedProjectionGpuOnly(input, qProj, input.hiddenStates, "gemma4-audio-attn-q-proj");
  const k = await runPackedProjectionGpuOnly(input, kProj, input.hiddenStates, "gemma4-audio-attn-k-proj");
  const v = await runPackedProjectionGpuOnly(input, vProj, input.hiddenStates, "gemma4-audio-attn-v-proj");
  const relativeKeys = await runF32LinearRowsWebGpu(input.device, {
    input: audioRelativePositionEmbeddings(input.positionLength, input.hiddenSize),
    weights: float32FromBytes(relativeKeyProjectionData.bytes),
    rows: input.positionLength,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    label: "gemma4-audio-attn-relative-k-proj",
  });
  const body = await runAudioSelfAttentionBodyWebGpu(input.device, {
    query: q,
    key: k,
    value: v,
    relativeKey: relativeKeys,
    perDimScale: float32FromBytes(perDimScaleData.bytes),
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    heads: input.heads,
    headDim: input.headDim,
    chunkSize: input.chunkSize,
    contextSize: input.contextSize,
    positionLength: input.positionLength,
    softcap: input.softcap,
    label: "gemma4-audio-attn-body",
  });
  return runPackedProjectionGpuOnly(input, postProj, body, "gemma4-audio-attn-post-proj");
}

export async function runPackedProjectionGpuOnly(
  input: {
    device: GPUDevice;
    rows: number;
    hiddenSize: number;
  },
  projection: PackedQatProjectionData<string>,
  gpuInput: Float32Array,
  label: string,
): Promise<Float32Array> {
  return runPackedQatLinearWebGpu(input.device, {
    input: gpuInput,
    weights: projection.weightData.bytes,
    weightScales: float32FromBytes(projection.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    bits: projection.bits,
    inputActivationScale: projection.inputActivationScale,
    outputActivationScale: projection.outputActivationScale,
    label,
  });
}

export function stripRowsByMask(values: Float32Array, mask: Uint8Array, dim: number): Float32Array {
  if (values.length !== mask.length * dim) {
    throw new Error("Masked row input length does not match dimensions.");
  }
  const rows = countMask(mask);
  const output = new Float32Array(rows * dim);
  let targetRow = 0;
  for (let row = 0; row < mask.length; row += 1) {
    if (mask[row] === 0) continue;
    output.set(values.subarray(row * dim, (row + 1) * dim), targetRow * dim);
    targetRow += 1;
  }
  return output;
}

export async function loadAudioSelfAttentionProjection(
  input: {
    urls: Gemma4MediaKernelProbe["urls"];
    header: SafetensorsHeader;
    layerIndex: number;
    signal?: AbortSignal;
  },
  projection: Gemma4AudioSelfAttentionProjectionName,
  inputDim: number,
  outputDim: number,
): Promise<PackedQatProjectionData<Gemma4AudioSelfAttentionProjectionName>> {
  return loadPackedQatProjection({
    urls: input.urls,
    header: input.header,
    names: audioSelfAttentionProjectionTensors(input.layerIndex, projection),
    projection,
    inputDim,
    outputDim,
    signal: input.signal,
  });
}

export async function runPackedProjectionPair(
  input: {
    device: GPUDevice;
    rows: number;
    hiddenSize: number;
  },
  projection: PackedQatProjectionData<string>,
  cpuInput: Float32Array,
  label: string,
  gpuInput = cpuInput,
): Promise<{ cpuOutput: Float32Array; gpuOutput: Float32Array }> {
  const weightScales = float32FromBytes(projection.weightScaleData.bytes);
  const cpuOutput = runPackedQatLinearCpu({
    input: cpuInput,
    weights: projection.weightData.bytes,
    weightScales,
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    bits: projection.bits,
    inputActivationScale: projection.inputActivationScale,
    outputActivationScale: projection.outputActivationScale,
  });
  const gpuOutput = await runPackedQatLinearWebGpu(input.device, {
    input: gpuInput,
    weights: projection.weightData.bytes,
    weightScales,
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    bits: projection.bits,
    inputActivationScale: projection.inputActivationScale,
    outputActivationScale: projection.outputActivationScale,
    label,
  });
  return { cpuOutput, gpuOutput };
}