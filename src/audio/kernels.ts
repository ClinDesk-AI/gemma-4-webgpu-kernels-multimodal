import { alignedBufferSize, shaderCompilationMessages, writeGpuBuffer } from "../gpu/runtime.js";
import { assertF32Tensor, float32FromBytes, loadSafetensorsTensorData } from "../io/safetensors.js";
import { runF32LinearRowsCpu, runF32LinearRowsWebGpu } from "../kernels/qat.js";
import { GPU_BUFFER_USAGE_COPY_DST, GPU_BUFFER_USAGE_COPY_SRC, GPU_BUFFER_USAGE_MAP_READ, GPU_BUFFER_USAGE_STORAGE, GPU_BUFFER_USAGE_UNIFORM, GPU_MAP_MODE_READ, audioOutputProjectionTensors } from "../model.js";
import { AudioSelfAttentionBodyInput, AudioSubsampleConvLayerInput, AudioSubsampleConvLayerOutput, AudioSubsampleProjectionInput, AudioSubsampleProjectionOutput, Gemma4MediaKernelProbe, SafetensorsHeader } from "../types.js";

export function computeGemma4AudioSoftTokensFromMask(mask: Uint8Array, audioSeqLength: number): number {
  let current = mask;
  for (let layer = 0; layer < 2; layer += 1) {
    const outputLength = Math.floor((current.length + 2 - 3) / 2) + 1;
    const next = new Uint8Array(Math.max(0, outputLength));
    for (let index = 0; index < next.length; index += 1) {
      next[index] = current[index * 2] ?? 0;
    }
    current = next;
  }
  let total = 0;
  for (const value of current) total += value;
  return Math.min(total, audioSeqLength);
}

export function runAudioSubsampleProjectionCpu(input: AudioSubsampleProjectionInput): AudioSubsampleProjectionOutput {
  const layer0 = runAudioSubsampleConvLayerCpu({
    inputValues: input.inputFeatures,
    inputMask: input.inputFeaturesMask,
    inputTime: input.frames,
    inputFreq: input.featureSize,
    inputChannels: 1,
    weights: input.weights.layer0Conv,
    normWeights: input.weights.layer0Norm,
    outputChannels: 128,
    epsilon: input.epsilon,
  });
  const layer1 = runAudioSubsampleConvLayerCpu({
    inputValues: layer0.output,
    inputMask: layer0.mask,
    inputTime: layer0.outputTime,
    inputFreq: layer0.outputFreq,
    inputChannels: 128,
    weights: input.weights.layer1Conv,
    normWeights: input.weights.layer1Norm,
    outputChannels: 32,
    epsilon: input.epsilon,
  });
  const rows = layer1.outputTime;
  const flattenedDim = layer1.outputFreq * 32;
  if (flattenedDim !== input.hiddenSize) {
    throw new Error(`Audio subsample flattened dim ${flattenedDim} does not match hidden size ${input.hiddenSize}.`);
  }
  return {
    output: runF32LinearRowsCpu(
      layer1.output,
      input.weights.inputProjection,
      rows,
      input.hiddenSize,
      input.hiddenSize,
    ),
    mask: layer1.mask,
    rows,
    validRows: countMask(layer1.mask),
  };
}

export async function runAudioSubsampleProjectionWebGpu(
  device: GPUDevice,
  input: AudioSubsampleProjectionInput,
): Promise<AudioSubsampleProjectionOutput> {
  const layer0 = await runAudioSubsampleConvLayerWebGpu(device, {
    inputValues: input.inputFeatures,
    inputMask: input.inputFeaturesMask,
    inputTime: input.frames,
    inputFreq: input.featureSize,
    inputChannels: 1,
    weights: input.weights.layer0Conv,
    normWeights: input.weights.layer0Norm,
    outputChannels: 128,
    epsilon: input.epsilon,
    label: "gemma4-audio-subsample-layer0",
  });
  const layer1 = await runAudioSubsampleConvLayerWebGpu(device, {
    inputValues: layer0.output,
    inputMask: layer0.mask,
    inputTime: layer0.outputTime,
    inputFreq: layer0.outputFreq,
    inputChannels: 128,
    weights: input.weights.layer1Conv,
    normWeights: input.weights.layer1Norm,
    outputChannels: 32,
    epsilon: input.epsilon,
    label: "gemma4-audio-subsample-layer1",
  });
  const rows = layer1.outputTime;
  const flattenedDim = layer1.outputFreq * 32;
  if (flattenedDim !== input.hiddenSize) {
    throw new Error(`Audio subsample flattened dim ${flattenedDim} does not match hidden size ${input.hiddenSize}.`);
  }
  return {
    output: await runF32LinearRowsWebGpu(device, {
      input: layer1.output,
      weights: input.weights.inputProjection,
      rows,
      inputDim: input.hiddenSize,
      outputDim: input.hiddenSize,
      label: "gemma4-audio-subsample-input-proj",
    }),
    mask: layer1.mask,
    rows,
    validRows: countMask(layer1.mask),
  };
}

export function runAudioSubsampleConvLayerCpu(input: AudioSubsampleConvLayerInput): AudioSubsampleConvLayerOutput {
  const outputTime = audioSubsampleOutputLength(input.inputTime);
  const outputFreq = audioSubsampleOutputLength(input.inputFreq);
  if (input.inputValues.length !== input.inputTime * input.inputFreq * input.inputChannels) {
    throw new Error("Audio subsample input length does not match dimensions.");
  }
  if (input.inputMask.length !== input.inputTime) {
    throw new Error("Audio subsample mask length does not match input time.");
  }
  if (input.weights.length !== input.outputChannels * input.inputChannels * 3 * 3) {
    throw new Error("Audio subsample conv weight length does not match dimensions.");
  }
  if (input.normWeights.length !== input.outputChannels) {
    throw new Error("Audio subsample norm weight length does not match output channels.");
  }
  const output = new Float32Array(outputTime * outputFreq * input.outputChannels);
  const channelValues = new Float32Array(input.outputChannels);
  for (let time = 0; time < outputTime; time += 1) {
    for (let freq = 0; freq < outputFreq; freq += 1) {
      let mean = 0;
      for (let channel = 0; channel < input.outputChannels; channel += 1) {
        const value = audioSubsampleConvValueCpu(input, time, freq, channel);
        channelValues[channel] = value;
        mean += value;
      }
      mean /= input.outputChannels;
      let variance = 0;
      for (let channel = 0; channel < input.outputChannels; channel += 1) {
        const centered = channelValues[channel] - mean;
        variance += centered * centered;
      }
      const invStd = (variance / input.outputChannels + input.epsilon) ** -0.5;
      for (let channel = 0; channel < input.outputChannels; channel += 1) {
        const normalized = (channelValues[channel] - mean) * invStd * input.normWeights[channel];
        output[(time * outputFreq + freq) * input.outputChannels + channel] = Math.max(0, normalized);
      }
    }
  }
  return {
    output,
    mask: strideAudioMask(input.inputMask, outputTime),
    outputTime,
    outputFreq,
  };
}

export function audioSubsampleConvValueCpu(
  input: AudioSubsampleConvLayerInput,
  outputTime: number,
  outputFreq: number,
  outputChannel: number,
): number {
  let sum = 0;
  for (let inputChannel = 0; inputChannel < input.inputChannels; inputChannel += 1) {
    for (let kernelTime = 0; kernelTime < 3; kernelTime += 1) {
      const sourceTime = outputTime * 2 + kernelTime - 1;
      if (sourceTime < 0 || sourceTime >= input.inputTime || input.inputMask[sourceTime] === 0) continue;
      for (let kernelFreq = 0; kernelFreq < 3; kernelFreq += 1) {
        const sourceFreq = outputFreq * 2 + kernelFreq - 1;
        if (sourceFreq < 0 || sourceFreq >= input.inputFreq) continue;
        const inputValue = input.inputValues[
          (sourceTime * input.inputFreq + sourceFreq) * input.inputChannels + inputChannel
        ];
        const weight = input.weights[
          ((outputChannel * input.inputChannels + inputChannel) * 3 + kernelTime) * 3 + kernelFreq
        ];
        sum += inputValue * weight;
      }
    }
  }
  return sum;
}

export async function runAudioSubsampleConvLayerWebGpu(
  device: GPUDevice,
  input: AudioSubsampleConvLayerInput & { label: string },
): Promise<AudioSubsampleConvLayerOutput> {
  const outputTime = audioSubsampleOutputLength(input.inputTime);
  const outputFreq = audioSubsampleOutputLength(input.inputFreq);
  const outputBytes = outputTime * outputFreq * input.outputChannels * 4;
  const maskValues = new Uint32Array(input.inputMask.length);
  for (let index = 0; index < input.inputMask.length; index += 1) maskValues[index] = input.inputMask[index];
  const inputBuffer = device.createBuffer({
    label: `${input.label}-input`,
    size: alignedBufferSize(input.inputValues.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const maskBuffer = device.createBuffer({
    label: `${input.label}-mask`,
    size: alignedBufferSize(maskValues.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightBuffer = device.createBuffer({
    label: `${input.label}-weights`,
    size: alignedBufferSize(input.weights.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const normBuffer = device.createBuffer({
    label: `${input.label}-norm`,
    size: alignedBufferSize(input.normWeights.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: `${input.label}-output`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const paramsBuffer = device.createBuffer({
    label: `${input.label}-params`,
    size: 64,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: `${input.label}-readback`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, inputBuffer, 0, input.inputValues);
    writeGpuBuffer(device, maskBuffer, 0, maskValues);
    writeGpuBuffer(device, weightBuffer, 0, input.weights);
    writeGpuBuffer(device, normBuffer, 0, input.normWeights);
    writeGpuBuffer(device, paramsBuffer, 0, new Uint32Array([
      input.inputTime,
      input.inputFreq,
      input.inputChannels,
      outputTime,
      outputFreq,
      input.outputChannels,
      outputTime * outputFreq * input.outputChannels,
      0,
    ]));
    writeGpuBuffer(device, paramsBuffer, 32, new Float32Array([input.epsilon, 0, 0, 0]));

    const module = device.createShaderModule({
      label: input.label,
      code: `
struct Params {
  inputTime: u32,
  inputFreq: u32,
  inputChannels: u32,
  outputTime: u32,
  outputFreq: u32,
  outputChannels: u32,
  totalValues: u32,
  _pad: u32,
  epsilon: f32,
  _pad2: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> inputMask: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> normWeights: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn sourceValue(time: i32, freq: i32, channel: u32) -> f32 {
  if (time < 0 || freq < 0) {
    return 0.0;
  }
  let sourceTime = u32(time);
  let sourceFreq = u32(freq);
  if (sourceTime >= params.inputTime || sourceFreq >= params.inputFreq || inputMask[sourceTime] == 0u) {
    return 0.0;
  }
  return inputValues[(sourceTime * params.inputFreq + sourceFreq) * params.inputChannels + channel];
}

fn convAt(outputTime: u32, outputFreq: u32, outputChannel: u32) -> f32 {
  var sum = 0.0;
  for (var inputChannel = 0u; inputChannel < params.inputChannels; inputChannel = inputChannel + 1u) {
    for (var kernelTime = 0u; kernelTime < 3u; kernelTime = kernelTime + 1u) {
      let sourceTime = i32(outputTime * 2u + kernelTime) - 1;
      for (var kernelFreq = 0u; kernelFreq < 3u; kernelFreq = kernelFreq + 1u) {
        let sourceFreq = i32(outputFreq * 2u + kernelFreq) - 1;
        let weight = weights[((outputChannel * params.inputChannels + inputChannel) * 3u + kernelTime) * 3u + kernelFreq];
        sum = sum + sourceValue(sourceTime, sourceFreq, inputChannel) * weight;
      }
    }
  }
  return sum;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let linear = id.x;
  if (linear >= params.totalValues) {
    return;
  }
  let channel = linear % params.outputChannels;
  let freq = (linear / params.outputChannels) % params.outputFreq;
  let time = linear / (params.outputChannels * params.outputFreq);

  var mean = 0.0;
  for (var c = 0u; c < params.outputChannels; c = c + 1u) {
    mean = mean + convAt(time, freq, c);
  }
  mean = mean / f32(params.outputChannels);

  var variance = 0.0;
  for (var c = 0u; c < params.outputChannels; c = c + 1u) {
    let centered = convAt(time, freq, c) - mean;
    variance = variance + centered * centered;
  }
  let invStd = inverseSqrt(variance / f32(params.outputChannels) + params.epsilon);
  let normalized = (convAt(time, freq, channel) - mean) * invStd * normWeights[channel];
  outputValues[linear] = max(normalized, 0.0);
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Audio subsample conv shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: input.label,
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: input.label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: maskBuffer } },
        { binding: 2, resource: { buffer: weightBuffer } },
        { binding: 3, resource: { buffer: normBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: input.label });
    const pass = encoder.beginComputePass({ label: input.label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((outputTime * outputFreq * input.outputChannels) / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(outputTime * outputFreq * input.outputChannels);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return {
      output,
      mask: strideAudioMask(input.inputMask, outputTime),
      outputTime,
      outputFreq,
    };
  } finally {
    inputBuffer.destroy();
    maskBuffer.destroy();
    weightBuffer.destroy();
    normBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export async function runAudioOutputProjectionWebGpu(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  hiddenStates: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const names = audioOutputProjectionTensors();
  const [weightData, biasData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.weight, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, names.bias, input.signal),
  ]);
  assertF32Tensor(weightData, [input.outputDim, input.inputDim], "audio output projection weight");
  assertF32Tensor(biasData, [input.outputDim], "audio output projection bias");
  const output = await runF32LinearRowsWebGpu(input.device, {
    input: input.hiddenStates,
    weights: float32FromBytes(weightData.bytes),
    rows: input.rows,
    inputDim: input.inputDim,
    outputDim: input.outputDim,
    label: "gemma4-audio-output-proj",
  });
  const bias = float32FromBytes(biasData.bytes);
  for (let row = 0; row < input.rows; row += 1) {
    const rowOffset = row * input.outputDim;
    for (let dim = 0; dim < input.outputDim; dim += 1) {
      output[rowOffset + dim] = Math.fround(output[rowOffset + dim] + bias[dim]);
    }
  }
  return output;
}

export function audioRelativePositionEmbeddings(positionLength: number, hiddenSize: number): Float32Array {
  if (hiddenSize % 2 !== 0) throw new Error("Audio relative position hidden size must be even.");
  const half = hiddenSize / 2;
  const output = new Float32Array(positionLength * hiddenSize);
  const logIncrement = Math.log(10000) / Math.max(half - 1, 1);
  for (let positionIndex = 0; positionIndex < positionLength; positionIndex += 1) {
    const position = positionLength - 1 - positionIndex;
    const offset = positionIndex * hiddenSize;
    for (let dim = 0; dim < half; dim += 1) {
      const scaled = position * Math.exp(dim * -logIncrement);
      output[offset + dim] = Math.sin(scaled);
      output[offset + half + dim] = Math.cos(scaled);
    }
  }
  return output;
}

export function runAudioSelfAttentionBodyCpu(input: AudioSelfAttentionBodyInput): Float32Array {
  const expectedLength = input.rows * input.hiddenSize;
  if (input.query.length !== expectedLength || input.key.length !== expectedLength || input.value.length !== expectedLength) {
    throw new Error("Audio self-attention q/k/v tensors do not match dimensions.");
  }
  if (input.relativeKey.length !== input.positionLength * input.hiddenSize) {
    throw new Error("Audio self-attention relative key tensor does not match dimensions.");
  }
  if (input.perDimScale.length !== input.headDim) {
    throw new Error("Audio self-attention per-dim scale length does not match head dimension.");
  }
  const output = new Float32Array(expectedLength);
  const scores = new Float32Array(input.contextSize);
  for (let row = 0; row < input.rows; row += 1) {
    for (let head = 0; head < input.heads; head += 1) {
      let maxScore = Number.NEGATIVE_INFINITY;
      for (let contextIndex = 0; contextIndex < input.contextSize; contextIndex += 1) {
        const score = audioSelfAttentionScore(input, row, head, contextIndex);
        scores[contextIndex] = score;
        maxScore = Math.max(maxScore, score);
      }
      let denominator = 0;
      for (let contextIndex = 0; contextIndex < input.contextSize; contextIndex += 1) {
        const weight = Math.fround(Math.exp(Math.fround(scores[contextIndex] - maxScore)));
        scores[contextIndex] = weight;
        denominator = Math.fround(denominator + weight);
      }
      for (let dim = 0; dim < input.headDim; dim += 1) {
        let sum = 0;
        for (let contextIndex = 0; contextIndex < input.contextSize; contextIndex += 1) {
          const sourceRow = audioSelfAttentionContextSourceRow(input, row, contextIndex);
          if (sourceRow < 0 || sourceRow >= input.rows) continue;
          const weight = Math.fround(scores[contextIndex] / denominator);
          const value = input.value[(sourceRow * input.heads + head) * input.headDim + dim];
          sum = Math.fround(sum + Math.fround(weight * value));
        }
        output[(row * input.heads + head) * input.headDim + dim] = sum;
      }
    }
  }
  return output;
}

export function audioSelfAttentionScore(
  input: AudioSelfAttentionBodyInput,
  row: number,
  head: number,
  contextIndex: number,
): number {
  const sourceRow = audioSelfAttentionContextSourceRow(input, row, contextIndex);
  const queryPos = row % input.chunkSize;
  const relIndex = contextIndex >= queryPos && contextIndex < queryPos + input.positionLength
    ? contextIndex - queryPos
    : -1;
  const qScale = (input.headDim ** -0.5) / Math.log(2);
  const kScale = Math.log(1 + Math.E) / Math.log(2);
  let score = 0;
  const queryBase = (row * input.heads + head) * input.headDim;
  for (let dim = 0; dim < input.headDim; dim += 1) {
    const query = input.query[queryBase + dim] * qScale * softplus(input.perDimScale[dim]);
    if (sourceRow >= 0 && sourceRow < input.rows) {
      const key = input.key[(sourceRow * input.heads + head) * input.headDim + dim] * kScale;
      score = Math.fround(score + Math.fround(query * key));
    }
    if (relIndex >= 0) {
      const relative = input.relativeKey[(relIndex * input.heads + head) * input.headDim + dim];
      score = Math.fround(score + Math.fround(query * relative));
    }
  }
  return Math.tanh(score / input.softcap) * input.softcap;
}

export function audioSelfAttentionContextSourceRow(
  input: Pick<AudioSelfAttentionBodyInput, "chunkSize" | "contextSize">,
  row: number,
  contextIndex: number,
): number {
  const block = Math.floor(row / input.chunkSize);
  const maxPastHorizon = input.contextSize - input.chunkSize;
  return block * input.chunkSize + contextIndex - maxPastHorizon;
}

export function softplus(value: number): number {
  if (value > 20) return value;
  if (value < -20) return Math.exp(value);
  return Math.log(1 + Math.exp(value));
}

export async function runAudioSelfAttentionBodyWebGpu(
  device: GPUDevice,
  input: AudioSelfAttentionBodyInput & { label: string },
): Promise<Float32Array> {
  const expectedLength = input.rows * input.hiddenSize;
  const outputBytes = expectedLength * 4;
  const queryBuffer = device.createBuffer({
    label: `${input.label}-query`,
    size: alignedBufferSize(input.query.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const keyBuffer = device.createBuffer({
    label: `${input.label}-key`,
    size: alignedBufferSize(input.key.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const valueBuffer = device.createBuffer({
    label: `${input.label}-value`,
    size: alignedBufferSize(input.value.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const relativeKeyBuffer = device.createBuffer({
    label: `${input.label}-relative-key`,
    size: alignedBufferSize(input.relativeKey.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const perDimScaleBuffer = device.createBuffer({
    label: `${input.label}-per-dim-scale`,
    size: alignedBufferSize(input.perDimScale.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: `${input.label}-output`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const paramsBuffer = device.createBuffer({
    label: `${input.label}-params`,
    size: 64,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: `${input.label}-readback`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, queryBuffer, 0, input.query);
    writeGpuBuffer(device, keyBuffer, 0, input.key);
    writeGpuBuffer(device, valueBuffer, 0, input.value);
    writeGpuBuffer(device, relativeKeyBuffer, 0, input.relativeKey);
    writeGpuBuffer(device, perDimScaleBuffer, 0, input.perDimScale);
    writeGpuBuffer(device, paramsBuffer, 0, new Uint32Array([
      input.rows,
      input.hiddenSize,
      input.heads,
      input.headDim,
      input.chunkSize,
      input.contextSize,
      input.positionLength,
      expectedLength,
    ]));
    writeGpuBuffer(device, paramsBuffer, 32, new Float32Array([input.softcap, 0, 0, 0]));

    const module = device.createShaderModule({
      label: input.label,
      code: `
struct Params {
  rows: u32,
  hiddenSize: u32,
  heads: u32,
  headDim: u32,
  chunkSize: u32,
  contextSize: u32,
  positionLength: u32,
  totalValues: u32,
  softcap: f32,
  _pad: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> queryValues: array<f32>;
@group(0) @binding(1) var<storage, read> keyValues: array<f32>;
@group(0) @binding(2) var<storage, read> valueValues: array<f32>;
@group(0) @binding(3) var<storage, read> relativeKeyValues: array<f32>;
@group(0) @binding(4) var<storage, read> perDimScaleValues: array<f32>;
@group(0) @binding(5) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

fn softplus(value: f32) -> f32 {
  if (value > 20.0) {
    return value;
  }
  if (value < -20.0) {
    return exp(value);
  }
  return log(1.0 + exp(value));
}

fn contextSourceRow(row: u32, contextIndex: u32) -> i32 {
  let block = row / params.chunkSize;
  let maxPastHorizon = params.contextSize - params.chunkSize;
  return i32(block * params.chunkSize + contextIndex) - i32(maxPastHorizon);
}

fn attentionScore(row: u32, head: u32, contextIndex: u32) -> f32 {
  let source = contextSourceRow(row, contextIndex);
  let queryPos = row % params.chunkSize;
  var relIndex = -1;
  if (contextIndex >= queryPos && contextIndex < queryPos + params.positionLength) {
    relIndex = i32(contextIndex - queryPos);
  }
  let qScale = inverseSqrt(f32(params.headDim)) / log(2.0);
  let kScale = log(1.0 + 2.718281828459045) / log(2.0);
  let queryBase = (row * params.heads + head) * params.headDim;
  var score = 0.0;
  for (var dim = 0u; dim < params.headDim; dim = dim + 1u) {
    let query = queryValues[queryBase + dim] * qScale * softplus(perDimScaleValues[dim]);
    if (source >= 0 && source < i32(params.rows)) {
      let key = keyValues[(u32(source) * params.heads + head) * params.headDim + dim] * kScale;
      score = score + query * key;
    }
    if (relIndex >= 0) {
      let relative = relativeKeyValues[(u32(relIndex) * params.heads + head) * params.headDim + dim];
      score = score + query * relative;
    }
  }
  return tanh(score / params.softcap) * params.softcap;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.totalValues) {
    return;
  }
  let dim = index % params.headDim;
  let head = (index / params.headDim) % params.heads;
  let row = index / params.hiddenSize;

  var maxScore = -3.4028234663852886e38;
  for (var contextIndex = 0u; contextIndex < params.contextSize; contextIndex = contextIndex + 1u) {
    maxScore = max(maxScore, attentionScore(row, head, contextIndex));
  }

  var denominator = 0.0;
  for (var contextIndex = 0u; contextIndex < params.contextSize; contextIndex = contextIndex + 1u) {
    denominator = denominator + exp(attentionScore(row, head, contextIndex) - maxScore);
  }

  var sum = 0.0;
  for (var contextIndex = 0u; contextIndex < params.contextSize; contextIndex = contextIndex + 1u) {
    let source = contextSourceRow(row, contextIndex);
    if (source >= 0 && source < i32(params.rows)) {
      let weight = exp(attentionScore(row, head, contextIndex) - maxScore) / denominator;
      let value = valueValues[(u32(source) * params.heads + head) * params.headDim + dim];
      sum = sum + weight * value;
    }
  }
  outputValues[index] = sum;
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Audio self-attention body shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: input.label,
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: input.label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: queryBuffer } },
        { binding: 1, resource: { buffer: keyBuffer } },
        { binding: 2, resource: { buffer: valueBuffer } },
        { binding: 3, resource: { buffer: relativeKeyBuffer } },
        { binding: 4, resource: { buffer: perDimScaleBuffer } },
        { binding: 5, resource: { buffer: outputBuffer } },
        { binding: 6, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: input.label });
    const pass = encoder.beginComputePass({ label: input.label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(expectedLength / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(expectedLength);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    queryBuffer.destroy();
    keyBuffer.destroy();
    valueBuffer.destroy();
    relativeKeyBuffer.destroy();
    perDimScaleBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export function runGluActivationCpu(values: Float32Array, rows: number, hiddenSize: number): Float32Array {
  if (values.length !== rows * hiddenSize * 2) {
    throw new Error("GLU input length does not match dimensions.");
  }
  const output = new Float32Array(rows * hiddenSize);
  for (let row = 0; row < rows; row += 1) {
    const inputOffset = row * hiddenSize * 2;
    const outputOffset = row * hiddenSize;
    for (let dim = 0; dim < hiddenSize; dim += 1) {
      const gate = values[inputOffset + dim];
      const sigmoid = 1 / (1 + Math.exp(-values[inputOffset + hiddenSize + dim]));
      output[outputOffset + dim] = Math.fround(gate * sigmoid);
    }
  }
  return output;
}

export async function runGluActivationWebGpu(
  device: GPUDevice,
  values: Float32Array,
  rows: number,
  hiddenSize: number,
  label: string,
): Promise<Float32Array> {
  if (values.length !== rows * hiddenSize * 2) {
    throw new Error("GLU WebGPU input length does not match dimensions.");
  }
  const outputBytes = rows * hiddenSize * 4;
  const inputBuffer = device.createBuffer({
    label: `${label}-input`,
    size: alignedBufferSize(values.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: `${label}-output`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: `${label}-dims`,
    size: 16,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: `${label}-readback`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, inputBuffer, 0, values);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([rows, hiddenSize, rows * hiddenSize, 0]));
    const module = device.createShaderModule({
      label,
      code: `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let totalValues = dims.z;
  if (index >= totalValues) {
    return;
  }
  let hiddenSize = dims.y;
  let row = index / hiddenSize;
  let dim = index % hiddenSize;
  let inputOffset = row * hiddenSize * 2u;
  let gate = inputValues[inputOffset + dim];
  let sig = 1.0 / (1.0 + exp(-inputValues[inputOffset + hiddenSize + dim]));
  outputValues[index] = gate * sig;
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`GLU shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: dimsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((rows * hiddenSize) / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(rows * hiddenSize);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    inputBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export function runAudioDepthwiseCausalConv1dCpu(input: {
  input: Float32Array;
  weights: Float32Array;
  rows: number;
  hiddenSize: number;
  kernelSize: number;
}): Float32Array {
  if (input.input.length !== input.rows * input.hiddenSize) {
    throw new Error("Audio depthwise conv input length does not match dimensions.");
  }
  if (input.weights.length !== input.hiddenSize * input.kernelSize) {
    throw new Error("Audio depthwise conv weight length does not match dimensions.");
  }
  const output = new Float32Array(input.input.length);
  const leftPad = input.kernelSize - 1;
  for (let row = 0; row < input.rows; row += 1) {
    for (let hidden = 0; hidden < input.hiddenSize; hidden += 1) {
      let sum = 0;
      for (let kernel = 0; kernel < input.kernelSize; kernel += 1) {
        const sourceRow = row + kernel - leftPad;
        if (sourceRow < 0 || sourceRow >= input.rows) continue;
        const value = input.input[sourceRow * input.hiddenSize + hidden];
        const weight = input.weights[hidden * input.kernelSize + kernel];
        sum = Math.fround(sum + Math.fround(value * weight));
      }
      output[row * input.hiddenSize + hidden] = sum;
    }
  }
  return output;
}

export async function runAudioDepthwiseCausalConv1dWebGpu(
  device: GPUDevice,
  input: {
    input: Float32Array;
    weights: Float32Array;
    rows: number;
    hiddenSize: number;
    kernelSize: number;
    label: string;
  },
): Promise<Float32Array> {
  if (input.input.length !== input.rows * input.hiddenSize) {
    throw new Error("Audio depthwise conv WebGPU input length does not match dimensions.");
  }
  if (input.weights.length !== input.hiddenSize * input.kernelSize) {
    throw new Error("Audio depthwise conv WebGPU weight length does not match dimensions.");
  }
  const outputBytes = input.input.byteLength;
  const inputBuffer = device.createBuffer({
    label: `${input.label}-input`,
    size: alignedBufferSize(input.input.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightBuffer = device.createBuffer({
    label: `${input.label}-weights`,
    size: alignedBufferSize(input.weights.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: `${input.label}-output`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: `${input.label}-dims`,
    size: 16,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: `${input.label}-readback`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, inputBuffer, 0, input.input);
    writeGpuBuffer(device, weightBuffer, 0, input.weights);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([
      input.rows,
      input.hiddenSize,
      input.kernelSize,
      input.rows * input.hiddenSize,
    ]));
    const module = device.createShaderModule({
      label: input.label,
      code: `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let rows = dims.x;
  let hiddenSize = dims.y;
  let kernelSize = dims.z;
  if (index >= dims.w) {
    return;
  }
  let hidden = index % hiddenSize;
  let row = index / hiddenSize;
  let leftPad = kernelSize - 1u;
  var sum = 0.0;
  for (var kernel = 0u; kernel < kernelSize; kernel = kernel + 1u) {
    let source = i32(row + kernel) - i32(leftPad);
    if (source >= 0 && source < i32(rows)) {
      sum = sum + inputValues[u32(source) * hiddenSize + hidden] * weights[hidden * kernelSize + kernel];
    }
  }
  outputValues[index] = sum;
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Audio depthwise conv shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: input.label,
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: input.label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: input.label });
    const pass = encoder.beginComputePass({ label: input.label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((input.rows * input.hiddenSize) / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(input.rows * input.hiddenSize);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    inputBuffer.destroy();
    weightBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export function runSiluActivationCpu(values: Float32Array): Float32Array {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = Math.fround(values[index] / (1 + Math.exp(-values[index])));
  }
  return output;
}

export async function runSiluActivationWebGpu(
  device: GPUDevice,
  values: Float32Array,
  label: string,
): Promise<Float32Array> {
  const outputBytes = values.byteLength;
  const inputBuffer = device.createBuffer({
    label: `${label}-input`,
    size: alignedBufferSize(values.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: `${label}-output`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: `${label}-dims`,
    size: 16,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: `${label}-readback`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, inputBuffer, 0, values);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([values.length, 0, 0, 0]));
    const module = device.createShaderModule({
      label,
      code: `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= dims.x) {
    return;
  }
  let value = inputValues[index];
  outputValues[index] = value / (1.0 + exp(-value));
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`SiLU shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: dimsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(values.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(values.length);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    inputBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export function runScaledResidualAddCpu(residual: Float32Array, values: Float32Array, scale: number): Float32Array {
  if (residual.length !== values.length) {
    throw new Error(`Scaled residual add length mismatch: ${residual.length} vs ${values.length}.`);
  }
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = Math.fround(residual[index] + Math.fround(values[index] * scale));
  }
  return output;
}

export async function runScaledResidualAddWebGpu(
  device: GPUDevice,
  residual: Float32Array,
  values: Float32Array,
  scale: number,
  label: string,
): Promise<Float32Array> {
  if (residual.length !== values.length) {
    throw new Error(`Scaled residual add WebGPU length mismatch: ${residual.length} vs ${values.length}.`);
  }
  const outputBytes = values.byteLength;
  const residualBuffer = device.createBuffer({
    label: `${label}-residual`,
    size: alignedBufferSize(residual.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const valuesBuffer = device.createBuffer({
    label: `${label}-values`,
    size: alignedBufferSize(values.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: `${label}-output`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const paramsBuffer = device.createBuffer({
    label: `${label}-params`,
    size: 64,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: `${label}-readback`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, residualBuffer, 0, residual);
    writeGpuBuffer(device, valuesBuffer, 0, values);
    writeGpuBuffer(device, paramsBuffer, 0, new Uint32Array([values.length, 0, 0, 0]));
    writeGpuBuffer(device, paramsBuffer, 16, new Float32Array([scale, 0, 0, 0]));
    const module = device.createShaderModule({
      label,
      code: `
struct Params {
  totalValues: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  scale: f32,
  _pad3: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> residualValues: array<f32>;
@group(0) @binding(1) var<storage, read> inputValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.totalValues) {
    return;
  }
  outputValues[index] = residualValues[index] + inputValues[index] * params.scale;
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Scaled residual add shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: residualBuffer } },
        { binding: 1, resource: { buffer: valuesBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(values.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(values.length);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    residualBuffer.destroy();
    valuesBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export function audioSubsampleOutputLength(inputLength: number): number {
  return Math.floor((inputLength + 2 - 3) / 2) + 1;
}

export function strideAudioMask(mask: Uint8Array, outputLength: number): Uint8Array {
  const output = new Uint8Array(outputLength);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = mask[index * 2] ?? 0;
  }
  return output;
}

export function countMask(mask: Uint8Array): number {
  let count = 0;
  for (const value of mask) count += value ? 1 : 0;
  return count;
}
