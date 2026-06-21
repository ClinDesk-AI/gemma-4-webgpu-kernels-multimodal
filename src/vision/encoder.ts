import { assertBf16Vector, float32FromBf16Bytes, float32FromBytes, int8FromBytes, loadSafetensorsTensorByteRange, loadSafetensorsTensorData, scalarF32FromTensor } from "../io/safetensors.js";
import { runQatI8LinearCpu, runQatI8LinearWebGpu } from "../kernels/qat.js";
import { checkedVisionLayerCount, visionAttentionNormTensor, visionAttentionOutputProjectionTensors, visionAttentionProjectionTensors, visionEncoderLayerNormTensors, visionMlpProjectionTensors } from "../model.js";
import { Gemma4MediaKernelProbe, Gemma4VisionAttentionProjectionName, Gemma4VisionEncoderStackKernelProbe, Gemma4VisionMlpProjectionName, Gemma4VisionMlpProjectionTensorSummary, SafetensorsHeader, SafetensorsTensorByteRange, SafetensorsTensorData } from "../types.js";
import { checksumBytes, checksumFloats, maxAbsDifference, maxAbsDifferenceStats, onesFloat32, roundedSample, visionRopePositions } from "../utils/math.js";
import { runResidualAddCpu, runResidualAddWebGpu, runVisionAttentionBodyCpu, runVisionAttentionBodyWebGpu, runVisionMlpActivationCpu, runVisionMlpActivationWebGpu, runVisionRmsNormCpu, runVisionRmsNormWebGpu, runVisionRopeCpu, runVisionRopeWebGpu } from "./kernels.js";

export async function loadVisionPatchPositionRows(input: {
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  tensorName: string;
  positions: Array<{ x: number; y: number }>;
  hiddenSize: number;
  signal?: AbortSignal;
}): Promise<{
  xRows: Float32Array;
  yRows: Float32Array;
  rowSlices: Array<{
    axis: "x" | "y";
    indexStart: number;
    rowCount: number;
    bytes: SafetensorsTensorByteRange;
  }>;
}> {
  const positionInfo = input.header.tensors[input.tensorName];
  if (!positionInfo) throw new Error(`Safetensors tensor not found: ${input.tensorName}`);
  if (positionInfo.dtype !== "BF16" ||
    positionInfo.shape.length !== 3 ||
    positionInfo.shape[0] !== 2 ||
    positionInfo.shape[2] !== input.hiddenSize
  ) {
    throw new Error(`Unexpected position table shape: ${positionInfo.dtype} [${positionInfo.shape.join(", ")}].`);
  }
  if (input.positions.length === 0) {
    throw new Error("Vision patch embedding requires at least one patch position.");
  }
  const maxX = Math.max(...input.positions.map((position) => position.x));
  const maxY = Math.max(...input.positions.map((position) => position.y));
  if (maxX < 0 || maxY < 0) {
    throw new Error("Vision patch positions must be non-negative.");
  }
  if (maxX >= positionInfo.shape[1] || maxY >= positionInfo.shape[1]) {
    throw new Error(
      `Vision patch positions exceed position table length ${positionInfo.shape[1]}: x=${maxX}, y=${maxY}.`,
    );
  }

  const rowBytes = input.hiddenSize * 2;
  const xRowCount = maxX + 1;
  const yRowCount = maxY + 1;
  const xBytes = await loadSafetensorsTensorByteRange(
    input.urls.safetensors,
    input.header,
    input.tensorName,
    0,
    xRowCount * rowBytes,
    input.signal,
  );
  const yBytes = await loadSafetensorsTensorByteRange(
    input.urls.safetensors,
    input.header,
    input.tensorName,
    positionInfo.shape[1] * rowBytes,
    yRowCount * rowBytes,
    input.signal,
  );
  return {
    xRows: float32FromBf16Bytes(xBytes.bytes),
    yRows: float32FromBf16Bytes(yBytes.bytes),
    rowSlices: [
      {
        axis: "x",
        indexStart: 0,
        rowCount: xRowCount,
        bytes: xBytes,
      },
      {
        axis: "y",
        indexStart: 0,
        rowCount: yRowCount,
        bytes: yBytes,
      },
    ],
  };
}

export async function computeVisionEncoderStack(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerCount: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  comparisonMode?: "propagated" | "gpuAnchored";
  positions?: Array<{ x: number; y: number }>;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<{
  layers: Gemma4VisionEncoderStackKernelProbe["layers"];
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
  maxAbsDiff: number;
}> {
  const layerCount = checkedVisionLayerCount(input.layerCount);
  const comparisonMode = input.comparisonMode ?? "propagated";
  let cpuHiddenStates = input.hiddenStates;
  let gpuHiddenStates = input.gpuHiddenStates ?? input.hiddenStates;
  const layers: Gemma4VisionEncoderStackKernelProbe["layers"] = [];

  for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
    if (input.signal?.aborted) throw new DOMException("Aborted", "AbortError");
    const cpuLayerInput = comparisonMode === "gpuAnchored" ? gpuHiddenStates : cpuHiddenStates;
    const layer = await computeVisionEncoderLayer({
      device: input.device,
      urls: input.urls,
      header: input.header,
      layerIndex,
      hiddenStates: cpuLayerInput,
      gpuHiddenStates,
      positions: input.positions,
      rows: input.rows,
      hiddenSize: input.hiddenSize,
      intermediateSize: input.intermediateSize,
      heads: input.heads,
      headDim: input.headDim,
      epsilon: input.epsilon,
      signal: input.signal,
    });
    cpuHiddenStates = layer.cpuOutput;
    gpuHiddenStates = layer.gpuOutput;
    const diffStats = maxAbsDifferenceStats(cpuHiddenStates, gpuHiddenStates, 0.01);
    layers.push({
      layerIndex,
      maxAbsDiff: layer.maxAbsDiff,
      maxAbsDiffIndex: diffStats.index,
      diffCountAboveTolerance: diffStats.countAboveTolerance,
      attentionMaxAbsDiff: layer.attentionMaxAbsDiff,
      mlpMaxAbsDiff: layer.mlpOutput.maxAbsDiff,
      attentionStageMaxAbsDiff: layer.attentionStageMaxAbsDiff,
      cpuChecksum: checksumFloats(cpuHiddenStates),
      gpuChecksum: checksumFloats(gpuHiddenStates),
      firstCpuValues: roundedSample(cpuHiddenStates),
      firstGpuValues: roundedSample(gpuHiddenStates),
      maxCpuValue: diffStats.cpuValue,
      maxGpuValue: diffStats.gpuValue,
    });
  }

  return {
    layers,
    cpuOutput: cpuHiddenStates,
    gpuOutput: gpuHiddenStates,
    maxAbsDiff: maxAbsDifference(cpuHiddenStates, gpuHiddenStates),
  };
}

export async function computeVisionEncoderStackWebGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerCount: number;
  hiddenStates: Float32Array;
  positions?: Array<{ x: number; y: number }>;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  let hiddenStates = input.hiddenStates;
  for (let layerIndex = 0; layerIndex < input.layerCount; layerIndex += 1) {
    if (input.signal?.aborted) throw new DOMException("Aborted", "AbortError");
    hiddenStates = await computeVisionEncoderLayerWebGpuOnly({
      ...input,
      layerIndex,
      hiddenStates,
    });
  }
  return hiddenStates;
}

export async function computeVisionEncoderLayerWebGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  hiddenStates: Float32Array;
  positions?: Array<{ x: number; y: number }>;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const normNames = visionEncoderLayerNormTensors(input.layerIndex);
  const [inputNormData, postAttentionNormData, preFeedForwardNormData, postFeedForwardNormData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.inputLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.postAttentionLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.preFeedForwardLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.postFeedForwardLayerNorm, input.signal),
  ]);
  assertBf16Vector(inputNormData, input.hiddenSize, `vision layer ${input.layerIndex} input layernorm`);
  assertBf16Vector(
    postAttentionNormData,
    input.hiddenSize,
    `vision layer ${input.layerIndex} post-attention layernorm`,
  );
  assertBf16Vector(
    preFeedForwardNormData,
    input.hiddenSize,
    `vision layer ${input.layerIndex} pre-feedforward layernorm`,
  );
  assertBf16Vector(
    postFeedForwardNormData,
    input.hiddenSize,
    `vision layer ${input.layerIndex} post-feedforward layernorm`,
  );

  const attentionInput = await runVisionRmsNormWebGpu(
    input.device,
    input.hiddenStates,
    float32FromBf16Bytes(inputNormData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const attentionOutput = await computeVisionAttentionOutputProjectionWebGpuOnly({
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    hiddenStates: attentionInput,
    positions: input.positions,
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    heads: input.heads,
    headDim: input.headDim,
    epsilon: input.epsilon,
    signal: input.signal,
  });
  const postAttention = await runVisionRmsNormWebGpu(
    input.device,
    attentionOutput,
    float32FromBf16Bytes(postAttentionNormData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const feedForwardResidual = await runResidualAddWebGpu(
    input.device,
    input.hiddenStates,
    postAttention,
    `gemma4-vision-layer-${input.layerIndex}-feedforward-residual`,
  );
  const feedForwardInput = await runVisionRmsNormWebGpu(
    input.device,
    feedForwardResidual,
    float32FromBf16Bytes(preFeedForwardNormData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const mlpOutput = await computeVisionMlpWebGpuOnly({
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    hiddenStates: feedForwardInput,
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    intermediateSize: input.intermediateSize,
    signal: input.signal,
  });
  const postFeedForward = await runVisionRmsNormWebGpu(
    input.device,
    mlpOutput,
    float32FromBf16Bytes(postFeedForwardNormData.bytes),
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  return runResidualAddWebGpu(
    input.device,
    feedForwardResidual,
    postFeedForward,
    `gemma4-vision-layer-${input.layerIndex}-output-residual`,
  );
}

export async function computeVisionAttentionProjection(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  projection: Gemma4VisionAttentionProjectionName;
  layerIndex?: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  signal?: AbortSignal;
}): Promise<{
  weightData: SafetensorsTensorData;
  weightScaleData: SafetensorsTensorData;
  inputActivationScale: number;
  outputActivationScale: number;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
  maxAbsDiff: number;
}> {
  const names = visionAttentionProjectionTensors(input.layerIndex ?? 0, input.projection);
  const [weightData, weightScaleData, inputScaleData, outputScaleData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weight, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weightScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.inputScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.outputScale, input.signal),
  ]);
  if (weightData.dtype !== "I8" ||
    weightData.shape.length !== 2 ||
    weightData.shape[0] !== input.outputDim ||
    weightData.shape[1] !== input.inputDim
  ) {
    throw new Error(`Unexpected ${input.projection} weight tensor: ${weightData.dtype} [${weightData.shape.join(", ")}].`);
  }
  if (weightScaleData.dtype !== "F32" ||
    weightScaleData.shape.length !== 2 ||
    weightScaleData.shape[0] !== input.outputDim ||
    weightScaleData.shape[1] !== 1
  ) {
    throw new Error(`Unexpected ${input.projection} weight scale tensor: ${weightScaleData.dtype} [${weightScaleData.shape.join(", ")}].`);
  }

  const inputActivationScale = scalarF32FromTensor(inputScaleData, names.inputScale);
  const outputActivationScale = scalarF32FromTensor(outputScaleData, names.outputScale);
  const weights = int8FromBytes(weightData.bytes);
  const weightScales = float32FromBytes(weightScaleData.bytes);
  const cpuOutput = runQatI8LinearCpu({
    input: input.hiddenStates,
    weights,
    weightScales,
    rows: input.rows,
    inputDim: input.inputDim,
    outputDim: input.outputDim,
    inputActivationScale,
    outputActivationScale,
  });
  const gpuOutput = await runQatI8LinearWebGpu(input.device, {
    input: input.gpuHiddenStates ?? input.hiddenStates,
    weights: weightData.bytes,
    weightScales,
    rows: input.rows,
    inputDim: input.inputDim,
    outputDim: input.outputDim,
    inputActivationScale,
    outputActivationScale,
    label: `gemma4-vision-${input.projection}`,
  });
  return {
    weightData,
    weightScaleData,
    inputActivationScale,
    outputActivationScale,
    cpuOutput,
    gpuOutput,
    maxAbsDiff: maxAbsDifference(cpuOutput, gpuOutput),
  };
}

export async function computeVisionAttentionProjectionWebGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  projection: Gemma4VisionAttentionProjectionName;
  layerIndex?: number;
  hiddenStates: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const names = visionAttentionProjectionTensors(input.layerIndex ?? 0, input.projection);
  const [weightData, weightScaleData, inputScaleData, outputScaleData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weight, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weightScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.inputScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.outputScale, input.signal),
  ]);
  if (weightData.dtype !== "I8" ||
    weightData.shape.length !== 2 ||
    weightData.shape[0] !== input.outputDim ||
    weightData.shape[1] !== input.inputDim
  ) {
    throw new Error(`Unexpected ${input.projection} weight tensor: ${weightData.dtype} [${weightData.shape.join(", ")}].`);
  }
  if (weightScaleData.dtype !== "F32" ||
    weightScaleData.shape.length !== 2 ||
    weightScaleData.shape[0] !== input.outputDim ||
    weightScaleData.shape[1] !== 1
  ) {
    throw new Error(`Unexpected ${input.projection} weight scale tensor: ${weightScaleData.dtype} [${weightScaleData.shape.join(", ")}].`);
  }
  return runQatI8LinearWebGpu(input.device, {
    input: input.hiddenStates,
    weights: weightData.bytes,
    weightScales: float32FromBytes(weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.inputDim,
    outputDim: input.outputDim,
    inputActivationScale: scalarF32FromTensor(inputScaleData, names.inputScale),
    outputActivationScale: scalarF32FromTensor(outputScaleData, names.outputScale),
    label: `gemma4-vision-${input.projection}`,
  });
}

export async function computeVisionAttentionNorm(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  projection: Gemma4VisionAttentionProjectionName;
  layerIndex?: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<{
  normTensor: SafetensorsTensorData | null;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
  maxAbsDiff: number;
}> {
  const projection = await computeVisionAttentionProjection(input);
  const normTensorName = visionAttentionNormTensor(input.layerIndex ?? 0, input.projection);
  const normTensor = normTensorName
    ? await loadSafetensorsTensorData(input.urls.safetensors, input.header, normTensorName, input.signal)
    : null;
  let normWeights: Float32Array;
  if (normTensor) {
    if (normTensor.dtype !== "BF16" || normTensor.shape.length !== 1 || normTensor.shape[0] !== input.headDim) {
      throw new Error(`Unexpected ${input.projection} norm tensor: ${normTensor.dtype} [${normTensor.shape.join(", ")}].`);
    }
    normWeights = float32FromBf16Bytes(normTensor.bytes);
  } else {
    normWeights = onesFloat32(input.headDim);
  }
  const cpuOutput = runVisionRmsNormCpu(
    projection.cpuOutput,
    normWeights,
    input.rows * input.heads,
    input.headDim,
    input.epsilon,
  );
  const gpuOutput = await runVisionRmsNormWebGpu(
    input.device,
    projection.gpuOutput,
    normWeights,
    input.rows * input.heads,
    input.headDim,
    input.epsilon,
  );
  return {
    normTensor,
    cpuOutput,
    gpuOutput,
    maxAbsDiff: maxAbsDifference(cpuOutput, gpuOutput),
  };
}

export async function computeVisionAttentionNormWebGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  projection: Gemma4VisionAttentionProjectionName;
  layerIndex?: number;
  hiddenStates: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const projected = await computeVisionAttentionProjectionWebGpuOnly(input);
  const normTensorName = visionAttentionNormTensor(input.layerIndex ?? 0, input.projection);
  const normTensor = normTensorName
    ? await loadSafetensorsTensorData(input.urls.safetensors, input.header, normTensorName, input.signal)
    : null;
  const normWeights = normTensor ? float32FromBf16Bytes(normTensor.bytes) : onesFloat32(input.headDim);
  if (normTensor &&
    (normTensor.dtype !== "BF16" || normTensor.shape.length !== 1 || normTensor.shape[0] !== input.headDim)
  ) {
    throw new Error(`Unexpected ${input.projection} norm tensor: ${normTensor.dtype} [${normTensor.shape.join(", ")}].`);
  }
  return runVisionRmsNormWebGpu(
    input.device,
    projected,
    normWeights,
    input.rows * input.heads,
    input.headDim,
    input.epsilon,
  );
}

export async function computeVisionAttentionOutputProjection(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex?: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  positions?: Array<{ x: number; y: number }>;
  rows: number;
  inputDim: number;
  outputDim: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<{
  weightData: SafetensorsTensorData;
  weightScaleData: SafetensorsTensorData;
  inputActivationScale: number;
  outputActivationScale: number;
  stageMaxAbsDiff: {
    qNorm: number;
    kNorm: number;
    vNorm: number;
    qRope: number;
    kRope: number;
    body: number;
    outputProjection: number;
  };
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
  maxAbsDiff: number;
}> {
  const ropeTheta = 100;
  const scaling = 1;
  const positions = input.positions ?? visionRopePositions(input.rows);
  if (positions.length !== input.rows) {
    throw new Error("Vision attention position count does not match rows.");
  }
  const common = {
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    hiddenStates: input.hiddenStates,
    gpuHiddenStates: input.gpuHiddenStates,
    rows: input.rows,
    inputDim: input.inputDim,
    outputDim: input.outputDim,
    heads: input.heads,
    headDim: input.headDim,
    epsilon: input.epsilon,
    signal: input.signal,
  };
  const [queryNorm, keyNorm, valueNorm] = await Promise.all([
    computeVisionAttentionNorm({ ...common, projection: "q_proj" }),
    computeVisionAttentionNorm({ ...common, projection: "k_proj" }),
    computeVisionAttentionNorm({ ...common, projection: "v_proj" }),
  ]);
  const queryCpu = runVisionRopeCpu(queryNorm.cpuOutput, positions, input.rows, input.heads, input.headDim, ropeTheta);
  const keyCpu = runVisionRopeCpu(keyNorm.cpuOutput, positions, input.rows, input.heads, input.headDim, ropeTheta);
  const queryGpu = await runVisionRopeWebGpu(
    input.device,
    queryNorm.gpuOutput,
    positions,
    input.rows,
    input.heads,
    input.headDim,
    ropeTheta,
  );
  const keyGpu = await runVisionRopeWebGpu(
    input.device,
    keyNorm.gpuOutput,
    positions,
    input.rows,
    input.heads,
    input.headDim,
    ropeTheta,
  );
  const attentionCpu = runVisionAttentionBodyCpu({
    query: queryCpu,
    key: keyCpu,
    value: valueNorm.cpuOutput,
    rows: input.rows,
    heads: input.heads,
    headDim: input.headDim,
    scaling,
  });
  const attentionGpu = await runVisionAttentionBodyWebGpu(input.device, {
    query: queryGpu,
    key: keyGpu,
    value: valueNorm.gpuOutput,
    rows: input.rows,
    heads: input.heads,
    headDim: input.headDim,
    scaling,
  });
  const names = visionAttentionOutputProjectionTensors(input.layerIndex ?? 0);
  const [weightData, weightScaleData, inputScaleData, outputScaleData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weight, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weightScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.inputScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.outputScale, input.signal),
  ]);
  if (weightData.dtype !== "I8" ||
    weightData.shape.length !== 2 ||
    weightData.shape[0] !== input.outputDim ||
    weightData.shape[1] !== input.outputDim
  ) {
    throw new Error(`Unexpected attention o_proj weight tensor: ${weightData.dtype} [${weightData.shape.join(", ")}].`);
  }
  if (weightScaleData.dtype !== "F32" ||
    weightScaleData.shape.length !== 2 ||
    weightScaleData.shape[0] !== input.outputDim ||
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
    rows: input.rows,
    inputDim: input.outputDim,
    outputDim: input.outputDim,
    inputActivationScale,
    outputActivationScale,
  });
  const gpuOutput = await runQatI8LinearWebGpu(input.device, {
    input: attentionGpu,
    weights: weightData.bytes,
    weightScales,
    rows: input.rows,
    inputDim: input.outputDim,
    outputDim: input.outputDim,
    inputActivationScale,
    outputActivationScale,
    label: "gemma4-vision-attention-o-proj",
  });
  return {
    weightData,
    weightScaleData,
    inputActivationScale,
    outputActivationScale,
    stageMaxAbsDiff: {
      qNorm: queryNorm.maxAbsDiff,
      kNorm: keyNorm.maxAbsDiff,
      vNorm: valueNorm.maxAbsDiff,
      qRope: maxAbsDifference(queryCpu, queryGpu),
      kRope: maxAbsDifference(keyCpu, keyGpu),
      body: maxAbsDifference(attentionCpu, attentionGpu),
      outputProjection: maxAbsDifference(cpuOutput, gpuOutput),
    },
    cpuOutput,
    gpuOutput,
    maxAbsDiff: maxAbsDifference(cpuOutput, gpuOutput),
  };
}

export async function computeVisionAttentionOutputProjectionWebGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex?: number;
  hiddenStates: Float32Array;
  positions?: Array<{ x: number; y: number }>;
  rows: number;
  inputDim: number;
  outputDim: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const ropeTheta = 100;
  const positions = input.positions ?? visionRopePositions(input.rows);
  if (positions.length !== input.rows) {
    throw new Error("Vision attention position count does not match rows.");
  }
  const common = {
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    hiddenStates: input.hiddenStates,
    rows: input.rows,
    inputDim: input.inputDim,
    outputDim: input.outputDim,
    heads: input.heads,
    headDim: input.headDim,
    epsilon: input.epsilon,
    signal: input.signal,
  };
  const [queryNorm, keyNorm, valueNorm] = await Promise.all([
    computeVisionAttentionNormWebGpuOnly({ ...common, projection: "q_proj" }),
    computeVisionAttentionNormWebGpuOnly({ ...common, projection: "k_proj" }),
    computeVisionAttentionNormWebGpuOnly({ ...common, projection: "v_proj" }),
  ]);
  const [query, key] = await Promise.all([
    runVisionRopeWebGpu(input.device, queryNorm, positions, input.rows, input.heads, input.headDim, ropeTheta),
    runVisionRopeWebGpu(input.device, keyNorm, positions, input.rows, input.heads, input.headDim, ropeTheta),
  ]);
  const attention = await runVisionAttentionBodyWebGpu(input.device, {
    query,
    key,
    value: valueNorm,
    rows: input.rows,
    heads: input.heads,
    headDim: input.headDim,
    scaling: 1,
  });
  const names = visionAttentionOutputProjectionTensors(input.layerIndex ?? 0);
  const [weightData, weightScaleData, inputScaleData, outputScaleData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weight, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weightScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.inputScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.outputScale, input.signal),
  ]);
  if (weightData.dtype !== "I8" ||
    weightData.shape.length !== 2 ||
    weightData.shape[0] !== input.outputDim ||
    weightData.shape[1] !== input.outputDim
  ) {
    throw new Error(`Unexpected attention o_proj weight tensor: ${weightData.dtype} [${weightData.shape.join(", ")}].`);
  }
  if (weightScaleData.dtype !== "F32" ||
    weightScaleData.shape.length !== 2 ||
    weightScaleData.shape[0] !== input.outputDim ||
    weightScaleData.shape[1] !== 1
  ) {
    throw new Error(`Unexpected attention o_proj weight scale tensor: ${weightScaleData.dtype} [${weightScaleData.shape.join(", ")}].`);
  }
  return runQatI8LinearWebGpu(input.device, {
    input: attention,
    weights: weightData.bytes,
    weightScales: float32FromBytes(weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.outputDim,
    outputDim: input.outputDim,
    inputActivationScale: scalarF32FromTensor(inputScaleData, names.inputScale),
    outputActivationScale: scalarF32FromTensor(outputScaleData, names.outputScale),
    label: "gemma4-vision-attention-o-proj",
  });
}

export async function computeVisionEncoderLayer(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  positions?: Array<{ x: number; y: number }>;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  heads: number;
  headDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<{
  inputNormData: SafetensorsTensorData;
  postAttentionNormData: SafetensorsTensorData;
  preFeedForwardNormData: SafetensorsTensorData;
  postFeedForwardNormData: SafetensorsTensorData;
  mlpOutput: Awaited<ReturnType<typeof computeVisionMlp>>;
  attentionMaxAbsDiff: number;
  attentionStageMaxAbsDiff: Awaited<ReturnType<typeof computeVisionAttentionOutputProjection>>["stageMaxAbsDiff"];
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
  maxAbsDiff: number;
}> {
  const normNames = visionEncoderLayerNormTensors(input.layerIndex);
  const [inputNormData, postAttentionNormData, preFeedForwardNormData, postFeedForwardNormData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.inputLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.postAttentionLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.preFeedForwardLayerNorm, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, normNames.postFeedForwardLayerNorm, input.signal),
  ]);
  assertBf16Vector(inputNormData, input.hiddenSize, `vision layer ${input.layerIndex} input layernorm`);
  assertBf16Vector(
    postAttentionNormData,
    input.hiddenSize,
    `vision layer ${input.layerIndex} post-attention layernorm`,
  );
  assertBf16Vector(
    preFeedForwardNormData,
    input.hiddenSize,
    `vision layer ${input.layerIndex} pre-feedforward layernorm`,
  );
  assertBf16Vector(
    postFeedForwardNormData,
    input.hiddenSize,
    `vision layer ${input.layerIndex} post-feedforward layernorm`,
  );

  const inputNormWeights = float32FromBf16Bytes(inputNormData.bytes);
  const postAttentionNormWeights = float32FromBf16Bytes(postAttentionNormData.bytes);
  const preFeedForwardNormWeights = float32FromBf16Bytes(preFeedForwardNormData.bytes);
  const postFeedForwardNormWeights = float32FromBf16Bytes(postFeedForwardNormData.bytes);
  const attentionInputCpu = runVisionRmsNormCpu(
    input.hiddenStates,
    inputNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const attentionInputGpu = await runVisionRmsNormWebGpu(
    input.device,
    input.gpuHiddenStates ?? input.hiddenStates,
    inputNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const attentionOutput = await computeVisionAttentionOutputProjection({
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    hiddenStates: attentionInputCpu,
    gpuHiddenStates: attentionInputGpu,
    positions: input.positions,
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.hiddenSize,
    heads: input.heads,
    headDim: input.headDim,
    epsilon: input.epsilon,
    signal: input.signal,
  });
  const postAttentionCpu = runVisionRmsNormCpu(
    attentionOutput.cpuOutput,
    postAttentionNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const postAttentionGpu = await runVisionRmsNormWebGpu(
    input.device,
    attentionOutput.gpuOutput,
    postAttentionNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const feedForwardResidualCpu = runResidualAddCpu(input.hiddenStates, postAttentionCpu);
  const feedForwardResidualGpu = await runResidualAddWebGpu(
    input.device,
    input.gpuHiddenStates ?? input.hiddenStates,
    postAttentionGpu,
    `gemma4-vision-layer-${input.layerIndex}-feedforward-residual`,
  );
  const feedForwardInputCpu = runVisionRmsNormCpu(
    feedForwardResidualCpu,
    preFeedForwardNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const feedForwardInputGpu = await runVisionRmsNormWebGpu(
    input.device,
    feedForwardResidualGpu,
    preFeedForwardNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const mlpOutput = await computeVisionMlp({
    device: input.device,
    urls: input.urls,
    header: input.header,
    layerIndex: input.layerIndex,
    hiddenStates: feedForwardInputCpu,
    gpuHiddenStates: feedForwardInputGpu,
    rows: input.rows,
    hiddenSize: input.hiddenSize,
    intermediateSize: input.intermediateSize,
    signal: input.signal,
  });
  const postFeedForwardCpu = runVisionRmsNormCpu(
    mlpOutput.cpuOutput,
    postFeedForwardNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const postFeedForwardGpu = await runVisionRmsNormWebGpu(
    input.device,
    mlpOutput.gpuOutput,
    postFeedForwardNormWeights,
    input.rows,
    input.hiddenSize,
    input.epsilon,
  );
  const cpuOutput = runResidualAddCpu(feedForwardResidualCpu, postFeedForwardCpu);
  const gpuOutput = await runResidualAddWebGpu(
    input.device,
    feedForwardResidualGpu,
    postFeedForwardGpu,
    `gemma4-vision-layer-${input.layerIndex}-output-residual`,
  );
  return {
    inputNormData,
    postAttentionNormData,
    preFeedForwardNormData,
    postFeedForwardNormData,
    mlpOutput,
    attentionMaxAbsDiff: attentionOutput.maxAbsDiff,
    attentionStageMaxAbsDiff: attentionOutput.stageMaxAbsDiff,
    cpuOutput,
    gpuOutput,
    maxAbsDiff: maxAbsDifference(cpuOutput, gpuOutput),
  };
}

export async function computeVisionMlp(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex?: number;
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  signal?: AbortSignal;
}): Promise<{
  projections: Array<{
    projection: Gemma4VisionMlpProjectionName;
    weightData: SafetensorsTensorData;
    weightScaleData: SafetensorsTensorData;
    inputActivationScale: number;
    outputActivationScale: number;
  }>;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
  maxAbsDiff: number;
}> {
  const [gate, up, down] = await Promise.all([
    loadVisionMlpProjection(input, "gate_proj", input.hiddenSize, input.intermediateSize),
    loadVisionMlpProjection(input, "up_proj", input.hiddenSize, input.intermediateSize),
    loadVisionMlpProjection(input, "down_proj", input.intermediateSize, input.hiddenSize),
  ]);
  const gateCpu = runQatI8LinearCpu({
    input: input.hiddenStates,
    weights: int8FromBytes(gate.weightData.bytes),
    weightScales: float32FromBytes(gate.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.intermediateSize,
    inputActivationScale: gate.inputActivationScale,
    outputActivationScale: gate.outputActivationScale,
  });
  const upCpu = runQatI8LinearCpu({
    input: input.hiddenStates,
    weights: int8FromBytes(up.weightData.bytes),
    weightScales: float32FromBytes(up.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.intermediateSize,
    inputActivationScale: up.inputActivationScale,
    outputActivationScale: up.outputActivationScale,
  });
  const gateGpu = await runQatI8LinearWebGpu(input.device, {
    input: input.gpuHiddenStates ?? input.hiddenStates,
    weights: gate.weightData.bytes,
    weightScales: float32FromBytes(gate.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.intermediateSize,
    inputActivationScale: gate.inputActivationScale,
    outputActivationScale: gate.outputActivationScale,
    label: "gemma4-vision-mlp-gate-proj",
  });
  const upGpu = await runQatI8LinearWebGpu(input.device, {
    input: input.gpuHiddenStates ?? input.hiddenStates,
    weights: up.weightData.bytes,
    weightScales: float32FromBytes(up.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.hiddenSize,
    outputDim: input.intermediateSize,
    inputActivationScale: up.inputActivationScale,
    outputActivationScale: up.outputActivationScale,
    label: "gemma4-vision-mlp-up-proj",
  });
  const activatedCpu = runVisionMlpActivationCpu(gateCpu, upCpu);
  const activatedGpu = await runVisionMlpActivationWebGpu(input.device, gateGpu, upGpu);
  const downWeightScales = float32FromBytes(down.weightScaleData.bytes);
  const cpuOutput = runQatI8LinearCpu({
    input: activatedCpu,
    weights: int8FromBytes(down.weightData.bytes),
    weightScales: downWeightScales,
    rows: input.rows,
    inputDim: input.intermediateSize,
    outputDim: input.hiddenSize,
    inputActivationScale: down.inputActivationScale,
    outputActivationScale: down.outputActivationScale,
  });
  const gpuOutput = await runQatI8LinearWebGpu(input.device, {
    input: activatedGpu,
    weights: down.weightData.bytes,
    weightScales: downWeightScales,
    rows: input.rows,
    inputDim: input.intermediateSize,
    outputDim: input.hiddenSize,
    inputActivationScale: down.inputActivationScale,
    outputActivationScale: down.outputActivationScale,
    label: "gemma4-vision-mlp-down-proj",
  });
  return {
    projections: [gate, up, down],
    cpuOutput,
    gpuOutput,
    maxAbsDiff: maxAbsDifference(cpuOutput, gpuOutput),
  };
}

export async function computeVisionMlpWebGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  layerIndex?: number;
  hiddenStates: Float32Array;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const [gate, up, down] = await Promise.all([
    loadVisionMlpProjection(input, "gate_proj", input.hiddenSize, input.intermediateSize),
    loadVisionMlpProjection(input, "up_proj", input.hiddenSize, input.intermediateSize),
    loadVisionMlpProjection(input, "down_proj", input.intermediateSize, input.hiddenSize),
  ]);
  const [gateGpu, upGpu] = await Promise.all([
    runQatI8LinearWebGpu(input.device, {
      input: input.hiddenStates,
      weights: gate.weightData.bytes,
      weightScales: float32FromBytes(gate.weightScaleData.bytes),
      rows: input.rows,
      inputDim: input.hiddenSize,
      outputDim: input.intermediateSize,
      inputActivationScale: gate.inputActivationScale,
      outputActivationScale: gate.outputActivationScale,
      label: "gemma4-vision-mlp-gate-proj",
    }),
    runQatI8LinearWebGpu(input.device, {
      input: input.hiddenStates,
      weights: up.weightData.bytes,
      weightScales: float32FromBytes(up.weightScaleData.bytes),
      rows: input.rows,
      inputDim: input.hiddenSize,
      outputDim: input.intermediateSize,
      inputActivationScale: up.inputActivationScale,
      outputActivationScale: up.outputActivationScale,
      label: "gemma4-vision-mlp-up-proj",
    }),
  ]);
  const activated = await runVisionMlpActivationWebGpu(input.device, gateGpu, upGpu);
  return runQatI8LinearWebGpu(input.device, {
    input: activated,
    weights: down.weightData.bytes,
    weightScales: float32FromBytes(down.weightScaleData.bytes),
    rows: input.rows,
    inputDim: input.intermediateSize,
    outputDim: input.hiddenSize,
    inputActivationScale: down.inputActivationScale,
    outputActivationScale: down.outputActivationScale,
    label: "gemma4-vision-mlp-down-proj",
  });
}

export async function loadVisionMlpProjection(
  input: {
    urls: Gemma4MediaKernelProbe["urls"];
    header: SafetensorsHeader;
    layerIndex?: number;
    signal?: AbortSignal;
  },
  projection: Gemma4VisionMlpProjectionName,
  inputDim: number,
  outputDim: number,
): Promise<{
  projection: Gemma4VisionMlpProjectionName;
  weightData: SafetensorsTensorData;
  weightScaleData: SafetensorsTensorData;
  inputActivationScale: number;
  outputActivationScale: number;
}> {
  const names = visionMlpProjectionTensors(input.layerIndex ?? 0, projection);
  const [weightData, weightScaleData, inputScaleData, outputScaleData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weight, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weightScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.inputScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.outputScale, input.signal),
  ]);
  if (weightData.dtype !== "I8" ||
    weightData.shape.length !== 2 ||
    weightData.shape[0] !== outputDim ||
    weightData.shape[1] !== inputDim
  ) {
    throw new Error(`Unexpected ${projection} weight tensor: ${weightData.dtype} [${weightData.shape.join(", ")}].`);
  }
  if (weightScaleData.dtype !== "F32" ||
    weightScaleData.shape.length !== 2 ||
    weightScaleData.shape[0] !== outputDim ||
    weightScaleData.shape[1] !== 1
  ) {
    throw new Error(`Unexpected ${projection} weight scale tensor: ${weightScaleData.dtype} [${weightScaleData.shape.join(", ")}].`);
  }
  return {
    projection,
    weightData,
    weightScaleData,
    inputActivationScale: scalarF32FromTensor(inputScaleData, names.inputScale),
    outputActivationScale: scalarF32FromTensor(outputScaleData, names.outputScale),
  };
}

export function visionMlpProjectionSummary(input: {
  projection: Gemma4VisionMlpProjectionName;
  weightData: SafetensorsTensorData;
  weightScaleData: SafetensorsTensorData;
  inputActivationScale: number;
  outputActivationScale: number;
}): Gemma4VisionMlpProjectionTensorSummary {
  return {
    projection: input.projection,
    weight: {
      name: input.weightData.name,
      dtype: input.weightData.dtype,
      shape: input.weightData.shape,
      dataBytes: input.weightData.dataBytes,
      fromCache: input.weightData.fromCache,
      checksum: checksumBytes(input.weightData.bytes),
    },
    weightScale: {
      name: input.weightScaleData.name,
      dtype: input.weightScaleData.dtype,
      shape: input.weightScaleData.shape,
      dataBytes: input.weightScaleData.dataBytes,
      fromCache: input.weightScaleData.fromCache,
      checksum: checksumBytes(input.weightScaleData.bytes),
    },
    inputActivationScale: input.inputActivationScale,
    outputActivationScale: input.outputActivationScale,
  };
}
