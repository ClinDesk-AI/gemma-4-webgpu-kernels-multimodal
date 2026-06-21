import { alignedBufferSize, shaderCompilationMessages, writeGpuBuffer } from "../gpu/runtime.js";
import { GPU_BUFFER_USAGE_COPY_DST, GPU_BUFFER_USAGE_COPY_SRC, GPU_BUFFER_USAGE_MAP_READ, GPU_BUFFER_USAGE_STORAGE, GPU_BUFFER_USAGE_UNIFORM, GPU_MAP_MODE_READ } from "../model.js";
import { VisionAttentionBodyInput, VisionPatchEmbeddingInput, VisionPoolerInput } from "../types.js";

export function runVisionRmsNormCpu(
  hiddenStates: Float32Array,
  weights: Float32Array,
  rows: number,
  hiddenSize: number,
  epsilon: number,
): Float32Array {
  if (hiddenStates.length !== rows * hiddenSize) {
    throw new Error("Vision RMSNorm hidden-state length does not match dimensions.");
  }
  if (weights.length !== hiddenSize) {
    throw new Error("Vision RMSNorm weight length does not match hidden size.");
  }
  const output = new Float32Array(hiddenStates.length);
  for (let row = 0; row < rows; row += 1) {
    const rowOffset = row * hiddenSize;
    let meanSquare = 0;
    for (let index = 0; index < hiddenSize; index += 1) {
      const value = hiddenStates[rowOffset + index];
      meanSquare += value * value;
    }
    const invRms = (meanSquare / hiddenSize + epsilon) ** -0.5;
    for (let index = 0; index < hiddenSize; index += 1) {
      output[rowOffset + index] = hiddenStates[rowOffset + index] * invRms * weights[index];
    }
  }
  return output;
}

export function runVisionRopeCpu(
  values: Float32Array,
  positions: Array<{ x: number; y: number }>,
  rows: number,
  heads: number,
  headDim: number,
  ropeTheta: number,
): Float32Array {
  if (values.length !== rows * heads * headDim) {
    throw new Error("Vision RoPE input length does not match dimensions.");
  }
  if (positions.length !== rows) {
    throw new Error("Vision RoPE position count does not match rows.");
  }
  if (headDim % 4 !== 0) {
    throw new Error("Vision RoPE head dimension must split into two even spatial halves.");
  }
  const output = new Float32Array(values.length);
  const partSize = headDim / 2;
  const halfPart = partSize / 2;
  for (let row = 0; row < rows; row += 1) {
    for (let head = 0; head < heads; head += 1) {
      const base = (row * heads + head) * headDim;
      for (let channel = 0; channel < headDim; channel += 1) {
        const part = channel < partSize ? 0 : 1;
        const partOffset = channel - part * partSize;
        const freqIndex = partOffset % halfPart;
        const position = part === 0 ? positions[row].x : positions[row].y;
        const angle = position / (ropeTheta ** ((2 * freqIndex) / partSize));
        const pairedOffset = partOffset < halfPart
          ? partOffset + halfPart
          : partOffset - halfPart;
        const rotated = (partOffset < halfPart ? -1 : 1) *
          values[base + part * partSize + pairedOffset];
        output[base + channel] = values[base + channel] * Math.cos(angle) + rotated * Math.sin(angle);
      }
    }
  }
  return output;
}

export function runVisionAttentionBodyCpu(input: VisionAttentionBodyInput): Float32Array {
  const expectedLength = input.rows * input.heads * input.headDim;
  if (input.query.length !== expectedLength ||
    input.key.length !== expectedLength ||
    input.value.length !== expectedLength
  ) {
    throw new Error("Vision attention body tensors do not match dimensions.");
  }
  const output = new Float32Array(expectedLength);
  const scores = new Float32Array(input.rows);
  for (let row = 0; row < input.rows; row += 1) {
    for (let head = 0; head < input.heads; head += 1) {
      let maxScore = Number.NEGATIVE_INFINITY;
      for (let keyRow = 0; keyRow < input.rows; keyRow += 1) {
        const score = Math.fround(
          visionAttentionScore(input.query, input.key, row, keyRow, head, input) * input.scaling,
        );
        scores[keyRow] = score;
        maxScore = Math.max(maxScore, score);
      }
      let denominator = 0;
      for (let keyRow = 0; keyRow < input.rows; keyRow += 1) {
        const weight = Math.fround(Math.exp(Math.fround(scores[keyRow] - maxScore)));
        scores[keyRow] = weight;
        denominator = Math.fround(denominator + weight);
      }
      const outputBase = (row * input.heads + head) * input.headDim;
      for (let dim = 0; dim < input.headDim; dim += 1) {
        let sum = 0;
        for (let keyRow = 0; keyRow < input.rows; keyRow += 1) {
          const valueBase = (keyRow * input.heads + head) * input.headDim;
          const weight = Math.fround(scores[keyRow] / denominator);
          sum = Math.fround(sum + Math.fround(weight * input.value[valueBase + dim]));
        }
        output[outputBase + dim] = sum;
      }
    }
  }
  return output;
}

export function visionAttentionScore(
  query: Float32Array,
  key: Float32Array,
  row: number,
  keyRow: number,
  head: number,
  input: Pick<VisionAttentionBodyInput, "heads" | "headDim">,
): number {
  const queryBase = (row * input.heads + head) * input.headDim;
  const keyBase = (keyRow * input.heads + head) * input.headDim;
  let score = 0;
  for (let dim = 0; dim < input.headDim; dim += 1) {
    score = Math.fround(score + Math.fround(query[queryBase + dim] * key[keyBase + dim]));
  }
  return score;
}

export function runResidualAddCpu(residual: Float32Array, values: Float32Array): Float32Array {
  if (residual.length !== values.length) {
    throw new Error(`Residual add length mismatch: ${residual.length} vs ${values.length}.`);
  }
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = residual[index] + values[index];
  }
  return output;
}

export function runVisionMlpActivationCpu(gate: Float32Array, up: Float32Array): Float32Array {
  if (gate.length !== up.length) {
    throw new Error(`MLP activation length mismatch: ${gate.length} vs ${up.length}.`);
  }
  const output = new Float32Array(gate.length);
  for (let index = 0; index < gate.length; index += 1) {
    output[index] = geluPytorchTanh(gate[index]) * up[index];
  }
  return output;
}

export function geluPytorchTanh(value: number): number {
  const inner = 0.7978845608028654 * (value + 0.044715 * value * value * value);
  const tanh = 2 / (1 + Math.exp(-2 * inner)) - 1;
  return 0.5 * value * (1 + tanh);
}

export async function runVisionRmsNormWebGpu(
  device: GPUDevice,
  hiddenStates: Float32Array,
  weights: Float32Array,
  rows: number,
  hiddenSize: number,
  epsilon: number,
): Promise<Float32Array> {
  const outputBytes = hiddenStates.byteLength;
  const hiddenBuffer = device.createBuffer({
    label: "gemma4-vision-rms-hidden",
    size: alignedBufferSize(hiddenStates.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightBuffer = device.createBuffer({
    label: "gemma4-vision-rms-weight",
    size: alignedBufferSize(weights.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: "gemma4-vision-rms-output",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: "gemma4-vision-rms-dims",
    size: 16,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: "gemma4-vision-rms-readback",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, hiddenBuffer, 0, hiddenStates);
    writeGpuBuffer(device, weightBuffer, 0, weights);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([
      rows,
      hiddenSize,
      Math.round(epsilon * 1_000_000_000),
      0,
    ]));

    const module = device.createShaderModule({
      label: "gemma4-vision-rmsnorm",
      code: `
@group(0) @binding(0) var<storage, read> hiddenStates: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let rows = dims.x;
  let hiddenSize = dims.y;
  let epsilon = f32(dims.z) / 1000000000.0;
  if (row >= rows) {
    return;
  }
  let rowOffset = row * hiddenSize;
  var meanSquare = 0.0;
  for (var index = 0u; index < hiddenSize; index = index + 1u) {
    let value = hiddenStates[rowOffset + index];
    meanSquare = meanSquare + value * value;
  }
  let invRms = inverseSqrt(meanSquare / f32(hiddenSize) + epsilon);
  for (var index = 0u; index < hiddenSize; index = index + 1u) {
    outputValues[rowOffset + index] = hiddenStates[rowOffset + index] * invRms * weights[index];
  }
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Vision RMSNorm shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "gemma4-vision-rmsnorm",
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: "gemma4-vision-rmsnorm",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: "gemma4-vision-rmsnorm" });
    const pass = encoder.beginComputePass({ label: "gemma4-vision-rmsnorm" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(rows);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(hiddenStates.length);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    hiddenBuffer.destroy();
    weightBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export async function runResidualAddWebGpu(
  device: GPUDevice,
  residual: Float32Array,
  values: Float32Array,
  label: string,
): Promise<Float32Array> {
  if (residual.length !== values.length) {
    throw new Error(`Residual add WebGPU length mismatch: ${residual.length} vs ${values.length}.`);
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
    writeGpuBuffer(device, residualBuffer, 0, residual);
    writeGpuBuffer(device, valuesBuffer, 0, values);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([values.length, 0, 0, 0]));

    const module = device.createShaderModule({
      label,
      code: `
@group(0) @binding(0) var<storage, read> residualValues: array<f32>;
@group(0) @binding(1) var<storage, read> inputValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= dims.x) {
    return;
  }
  outputValues[index] = residualValues[index] + inputValues[index];
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Residual add shader failed validation: ${compilationMessages.join("; ")}`);
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
        { binding: 3, resource: { buffer: dimsBuffer } },
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
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export async function runVisionMlpActivationWebGpu(
  device: GPUDevice,
  gate: Float32Array,
  up: Float32Array,
): Promise<Float32Array> {
  if (gate.length !== up.length) {
    throw new Error(`MLP activation WebGPU length mismatch: ${gate.length} vs ${up.length}.`);
  }
  const outputBytes = gate.byteLength;
  const gateBuffer = device.createBuffer({
    label: "gemma4-vision-mlp-activation-gate",
    size: alignedBufferSize(gate.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const upBuffer = device.createBuffer({
    label: "gemma4-vision-mlp-activation-up",
    size: alignedBufferSize(up.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: "gemma4-vision-mlp-activation-output",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: "gemma4-vision-mlp-activation-dims",
    size: 16,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: "gemma4-vision-mlp-activation-readback",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, gateBuffer, 0, gate);
    writeGpuBuffer(device, upBuffer, 0, up);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([gate.length, 0, 0, 0]));

    const module = device.createShaderModule({
      label: "gemma4-vision-mlp-activation",
      code: `
@group(0) @binding(0) var<storage, read> gateValues: array<f32>;
@group(0) @binding(1) var<storage, read> upValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

fn geluPytorchTanh(value: f32) -> f32 {
  let inner = 0.7978845608028654 * (value + 0.044715 * value * value * value);
  let tanhValue = 2.0 / (1.0 + exp(-2.0 * inner)) - 1.0;
  return 0.5 * value * (1.0 + tanhValue);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= dims.x) {
    return;
  }
  outputValues[index] = geluPytorchTanh(gateValues[index]) * upValues[index];
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Vision MLP activation shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "gemma4-vision-mlp-activation",
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: "gemma4-vision-mlp-activation",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gateBuffer } },
        { binding: 1, resource: { buffer: upBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: "gemma4-vision-mlp-activation" });
    const pass = encoder.beginComputePass({ label: "gemma4-vision-mlp-activation" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(gate.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(gate.length);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    gateBuffer.destroy();
    upBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export async function runVisionAttentionBodyWebGpu(
  device: GPUDevice,
  input: VisionAttentionBodyInput,
): Promise<Float32Array> {
  const expectedLength = input.rows * input.heads * input.headDim;
  if (input.query.length !== expectedLength ||
    input.key.length !== expectedLength ||
    input.value.length !== expectedLength
  ) {
    throw new Error("Vision attention body WebGPU tensors do not match dimensions.");
  }
  const outputBytes = expectedLength * 4;
  const queryBuffer = device.createBuffer({
    label: "gemma4-vision-attention-body-query",
    size: alignedBufferSize(input.query.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const keyBuffer = device.createBuffer({
    label: "gemma4-vision-attention-body-key",
    size: alignedBufferSize(input.key.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const valueBuffer = device.createBuffer({
    label: "gemma4-vision-attention-body-value",
    size: alignedBufferSize(input.value.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: "gemma4-vision-attention-body-output",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const paramsBuffer = device.createBuffer({
    label: "gemma4-vision-attention-body-params",
    size: 32,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: "gemma4-vision-attention-body-readback",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, queryBuffer, 0, input.query);
    writeGpuBuffer(device, keyBuffer, 0, input.key);
    writeGpuBuffer(device, valueBuffer, 0, input.value);
    writeGpuBuffer(device, paramsBuffer, 0, new Uint32Array([
      input.rows,
      input.heads,
      input.headDim,
      0,
    ]));
    writeGpuBuffer(device, paramsBuffer, 16, new Float32Array([
      input.scaling,
      0,
      0,
      0,
    ]));

    const module = device.createShaderModule({
      label: "gemma4-vision-attention-body",
      code: `
struct Params {
  rows: u32,
  heads: u32,
  headDim: u32,
  _pad: u32,
  scaling: f32,
  _pad2: f32,
  _pad3: f32,
  _pad4: f32,
};

@group(0) @binding(0) var<storage, read> queryValues: array<f32>;
@group(0) @binding(1) var<storage, read> keyValues: array<f32>;
@group(0) @binding(2) var<storage, read> valueValues: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn attentionScore(queryRow: u32, keyRow: u32, head: u32) -> f32 {
  let queryBase = (queryRow * params.heads + head) * params.headDim;
  let keyBase = (keyRow * params.heads + head) * params.headDim;
  var score = 0.0;
  for (var dim = 0u; dim < params.headDim; dim = dim + 1u) {
    score = score + queryValues[queryBase + dim] * keyValues[keyBase + dim];
  }
  return score * params.scaling;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let linear = id.x;
  let totalValues = params.rows * params.heads * params.headDim;
  if (linear >= totalValues) {
    return;
  }
  let dim = linear % params.headDim;
  let head = (linear / params.headDim) % params.heads;
  let row = linear / (params.heads * params.headDim);

  var maxScore = -3.4028234663852886e38;
  for (var keyRow = 0u; keyRow < params.rows; keyRow = keyRow + 1u) {
    maxScore = max(maxScore, attentionScore(row, keyRow, head));
  }

  var denominator = 0.0;
  for (var keyRow = 0u; keyRow < params.rows; keyRow = keyRow + 1u) {
    denominator = denominator + exp(attentionScore(row, keyRow, head) - maxScore);
  }

  var sum = 0.0;
  for (var keyRow = 0u; keyRow < params.rows; keyRow = keyRow + 1u) {
    let weight = exp(attentionScore(row, keyRow, head) - maxScore) / denominator;
    let valueBase = (keyRow * params.heads + head) * params.headDim;
    sum = sum + weight * valueValues[valueBase + dim];
  }
  outputValues[linear] = sum;
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Vision attention body shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "gemma4-vision-attention-body",
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: "gemma4-vision-attention-body",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: queryBuffer } },
        { binding: 1, resource: { buffer: keyBuffer } },
        { binding: 2, resource: { buffer: valueBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: "gemma4-vision-attention-body" });
    const pass = encoder.beginComputePass({ label: "gemma4-vision-attention-body" });
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
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export async function runVisionRopeWebGpu(
  device: GPUDevice,
  values: Float32Array,
  positions: Array<{ x: number; y: number }>,
  rows: number,
  heads: number,
  headDim: number,
  ropeTheta: number,
): Promise<Float32Array> {
  if (values.length !== rows * heads * headDim) {
    throw new Error("Vision RoPE WebGPU input length does not match dimensions.");
  }
  const positionValues = new Uint32Array(rows * 2);
  for (let row = 0; row < rows; row += 1) {
    positionValues[row * 2] = positions[row]?.x ?? 0;
    positionValues[row * 2 + 1] = positions[row]?.y ?? 0;
  }
  const outputBytes = values.byteLength;
  const inputBuffer = device.createBuffer({
    label: "gemma4-vision-rope-input",
    size: alignedBufferSize(values.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const positionBuffer = device.createBuffer({
    label: "gemma4-vision-rope-positions",
    size: alignedBufferSize(positionValues.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: "gemma4-vision-rope-output",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: "gemma4-vision-rope-dims",
    size: 16,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: "gemma4-vision-rope-readback",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, inputBuffer, 0, values);
    writeGpuBuffer(device, positionBuffer, 0, positionValues);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([
      rows,
      heads,
      headDim,
      Math.round(ropeTheta),
    ]));

    const module = device.createShaderModule({
      label: "gemma4-vision-rope",
      code: `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let linear = id.x;
  let rows = dims.x;
  let heads = dims.y;
  let headDim = dims.z;
  let totalValues = rows * heads * headDim;
  if (linear >= totalValues) {
    return;
  }
  let channel = linear % headDim;
  let head = (linear / headDim) % heads;
  let row = linear / (heads * headDim);
  let partSize = headDim / 2u;
  let halfPart = partSize / 2u;
  var part = 0u;
  if (channel >= partSize) {
    part = 1u;
  }
  let partOffset = channel - part * partSize;
  let freqIndex = partOffset % halfPart;
  let position = f32(positions[row * 2u + part]);
  let exponent = f32(freqIndex * 2u) / f32(partSize);
  let angle = position / pow(f32(dims.w), exponent);
  var pairedOffset = partOffset + halfPart;
  var rotateSign = -1.0;
  if (partOffset >= halfPart) {
    pairedOffset = partOffset - halfPart;
    rotateSign = 1.0;
  }
  let headBase = (row * heads + head) * headDim;
  let pairedValue = inputValues[headBase + part * partSize + pairedOffset];
  outputValues[linear] = inputValues[linear] * cos(angle) + rotateSign * pairedValue * sin(angle);
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Vision RoPE shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "gemma4-vision-rope",
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: "gemma4-vision-rope",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: positionBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: "gemma4-vision-rope" });
    const pass = encoder.beginComputePass({ label: "gemma4-vision-rope" });
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
    positionBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export function runVisionPoolerCpu(input: VisionPoolerInput): Float32Array {
  const poolIndices = visionPoolIndices(input);
  const kSquared = input.poolingKernelSize ** 2;
  const output = new Float32Array(input.outputLength * input.hiddenSize);
  const scale = Math.sqrt(input.hiddenSize);
  for (let token = 0; token < input.outputLength; token += 1) {
    for (let hidden = 0; hidden < input.hiddenSize; hidden += 1) {
      let sum = 0;
      for (let poolIndex = 0; poolIndex < kSquared; poolIndex += 1) {
        const patchIndex = poolIndices[token * kSquared + poolIndex];
        if (patchIndex >= 0) {
          sum += input.hiddenStates[patchIndex * input.hiddenSize + hidden];
        }
      }
      output[token * input.hiddenSize + hidden] = (sum / kSquared) * scale;
    }
  }
  return output;
}

export async function runVisionPoolerWebGpu(
  device: GPUDevice,
  input: VisionPoolerInput,
): Promise<Float32Array> {
  const poolIndices = visionPoolIndices(input);
  const kSquared = input.poolingKernelSize ** 2;
  const outputBytes = input.outputLength * input.hiddenSize * 4;
  const hiddenBuffer = device.createBuffer({
    label: "gemma4-vision-pooler-hidden",
    size: alignedBufferSize(input.hiddenStates.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const poolIndexBuffer = device.createBuffer({
    label: "gemma4-vision-pooler-indices",
    size: alignedBufferSize(poolIndices.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: "gemma4-vision-pooler-output",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: "gemma4-vision-pooler-dims",
    size: 16,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: "gemma4-vision-pooler-readback",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, hiddenBuffer, 0, input.hiddenStates);
    writeGpuBuffer(device, poolIndexBuffer, 0, poolIndices);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([
      input.outputLength,
      input.hiddenSize,
      kSquared,
      0,
    ]));

    const module = device.createShaderModule({
      label: "gemma4-vision-pooler",
      code: `
@group(0) @binding(0) var<storage, read> hiddenStates: array<f32>;
@group(0) @binding(1) var<storage, read> poolIndices: array<i32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let linear = id.x;
  let outputLength = dims.x;
  let hiddenSize = dims.y;
  let kSquared = dims.z;
  if (linear >= outputLength * hiddenSize) {
    return;
  }
  let hidden = linear % hiddenSize;
  let token = linear / hiddenSize;
  var sum = 0.0;
  for (var poolOffset = 0u; poolOffset < kSquared; poolOffset = poolOffset + 1u) {
    let patchIndex = poolIndices[token * kSquared + poolOffset];
    if (patchIndex >= 0) {
      sum = sum + hiddenStates[u32(patchIndex) * hiddenSize + hidden];
    }
  }
  outputValues[linear] = (sum / f32(kSquared)) * sqrt(f32(hiddenSize));
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Vision pooler shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "gemma4-vision-pooler",
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: "gemma4-vision-pooler",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: poolIndexBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: "gemma4-vision-pooler" });
    const pass = encoder.beginComputePass({ label: "gemma4-vision-pooler" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((input.outputLength * input.hiddenSize) / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(input.outputLength * input.hiddenSize);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    hiddenBuffer.destroy();
    poolIndexBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export function visionPoolIndices(input: VisionPoolerInput): Int32Array {
  if (input.hiddenStates.length !== input.positions.length * input.hiddenSize) {
    throw new Error("Vision pooler hidden-state length does not match positions.");
  }
  const k = input.poolingKernelSize;
  const kSquared = k ** 2;
  const maxX = Math.max(...input.positions.map((position) => position.x)) + 1;
  const maxY = Math.max(...input.positions.map((position) => position.y)) + 1;
  const cellsX = Math.floor(maxX / k);
  const cellsY = Math.floor(maxY / k);
  if (cellsX * cellsY !== input.outputLength) {
    throw new Error(`Vision pooler output length ${input.outputLength} does not match ${cellsX} x ${cellsY}.`);
  }
  const indices = new Int32Array(input.outputLength * kSquared);
  indices.fill(-1);
  for (let patchIndex = 0; patchIndex < input.positions.length; patchIndex += 1) {
    const position = input.positions[patchIndex];
    const token = Math.floor(position.x / k) + cellsX * Math.floor(position.y / k);
    const local = (position.y % k) * k + (position.x % k);
    if (token < 0 || token >= input.outputLength) {
      throw new Error(`Vision pooler patch position (${position.x}, ${position.y}) maps outside output length.`);
    }
    indices[token * kSquared + local] = patchIndex;
  }
  return indices;
}

export function runVisionPatchEmbeddingCpu(input: VisionPatchEmbeddingInput): Float32Array {
  if (input.projectionWeights.length !== input.patchPixels * input.hiddenSize) {
    throw new Error("Vision patch projection weight length does not match patch dimensions.");
  }
  if (input.patchValues.length !== input.positions.length * input.patchPixels) {
    throw new Error("Vision patch value length does not match patch count.");
  }
  if (input.outputRows <= 0 || input.outputRows > input.hiddenSize) {
    throw new Error("Vision patch output row count must fit in hidden size.");
  }
  if (input.positionXRows.length % input.hiddenSize !== 0 || input.positionYRows.length % input.hiddenSize !== 0) {
    throw new Error("Vision position rows must be a whole number of hidden-size rows.");
  }
  const output = new Float32Array(input.positions.length * input.outputRows);
  for (let patch = 0; patch < input.positions.length; patch += 1) {
    const position = input.positions[patch];
    const xOffset = position.x * input.hiddenSize;
    const yOffset = position.y * input.hiddenSize;
    if (xOffset < 0 || xOffset + input.hiddenSize > input.positionXRows.length) {
      throw new Error(`Vision patch x position ${position.x} is outside loaded position rows.`);
    }
    if (yOffset < 0 || yOffset + input.hiddenSize > input.positionYRows.length) {
      throw new Error(`Vision patch y position ${position.y} is outside loaded position rows.`);
    }
    for (let row = 0; row < input.outputRows; row += 1) {
      let sum = 0;
      const weightOffset = row * input.patchPixels;
      const patchOffset = patch * input.patchPixels;
      for (let col = 0; col < input.patchPixels; col += 1) {
        const scaledPixel = 2 * (input.patchValues[patchOffset + col] - 0.5);
        sum += scaledPixel * input.projectionWeights[weightOffset + col];
      }
      output[patch * input.outputRows + row] =
        sum + input.positionXRows[xOffset + row] + input.positionYRows[yOffset + row];
    }
  }
  return output;
}

export async function runVisionPatchEmbeddingWebGpu(
  device: GPUDevice,
  input: VisionPatchEmbeddingInput,
): Promise<Float32Array> {
  if (input.positionXRows.length % input.hiddenSize !== 0 || input.positionYRows.length % input.hiddenSize !== 0) {
    throw new Error("Vision position rows must be a whole number of hidden-size rows.");
  }
  const xRowCount = input.positionXRows.length / input.hiddenSize;
  const yRowCount = input.positionYRows.length / input.hiddenSize;
  const positionRows = new Float32Array(input.positionXRows.length + input.positionYRows.length);
  positionRows.set(input.positionXRows, 0);
  positionRows.set(input.positionYRows, input.positionXRows.length);
  const positionIndices = new Uint32Array(input.positions.length * 2);
  for (let index = 0; index < input.positions.length; index += 1) {
    if (input.positions[index].x < 0 || input.positions[index].x >= xRowCount) {
      throw new Error(`Vision patch x position ${input.positions[index].x} is outside loaded position rows.`);
    }
    if (input.positions[index].y < 0 || input.positions[index].y >= yRowCount) {
      throw new Error(`Vision patch y position ${input.positions[index].y} is outside loaded position rows.`);
    }
    positionIndices[index * 2] = input.positions[index].x;
    positionIndices[index * 2 + 1] = input.positions[index].y + xRowCount;
  }
  const outputBytes = input.positions.length * input.outputRows * 4;
  const patchBuffer = device.createBuffer({
    label: "gemma4-vision-patch-values",
    size: alignedBufferSize(input.patchValues.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightBuffer = device.createBuffer({
    label: "gemma4-vision-patch-projection",
    size: alignedBufferSize(input.projectionWeights.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const positionRowsBuffer = device.createBuffer({
    label: "gemma4-vision-position-rows",
    size: alignedBufferSize(positionRows.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const patchPositionsBuffer = device.createBuffer({
    label: "gemma4-vision-patch-positions",
    size: alignedBufferSize(positionIndices.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: "gemma4-vision-patch-output",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: "gemma4-vision-patch-dims",
    size: 32,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: "gemma4-vision-patch-readback",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, patchBuffer, 0, input.patchValues);
    writeGpuBuffer(device, weightBuffer, 0, input.projectionWeights);
    writeGpuBuffer(device, positionRowsBuffer, 0, positionRows);
    writeGpuBuffer(device, patchPositionsBuffer, 0, positionIndices);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([
      input.positions.length,
      input.patchPixels,
      input.outputRows,
      input.hiddenSize,
      0,
      0,
      0,
      0,
    ]));

    const module = device.createShaderModule({
      label: "gemma4-vision-patch-embedding",
      code: `
struct PatchDims {
  patchCount: u32,
  patchPixels: u32,
  outputRows: u32,
  hiddenSize: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<storage, read> patchValues: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> positionRows: array<f32>;
@group(0) @binding(3) var<storage, read> patchPositions: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(5) var<uniform> dims: PatchDims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let linear = id.x;
  let patchCount = dims.patchCount;
  let patchPixels = dims.patchPixels;
  let outputRows = dims.outputRows;
  let positionRowStride = dims.hiddenSize;
  if (linear >= patchCount * outputRows) {
    return;
  }
  let patchIndex = linear / outputRows;
  let row = linear % outputRows;
  var sum = 0.0;
  let patchOffset = patchIndex * patchPixels;
  let weightOffset = row * patchPixels;
  for (var col = 0u; col < patchPixels; col = col + 1u) {
    let scaledPixel = 2.0 * (patchValues[patchOffset + col] - 0.5);
    sum = sum + scaledPixel * weights[weightOffset + col];
  }
  let xRow = patchPositions[patchIndex * 2u];
  let yRow = patchPositions[patchIndex * 2u + 1u];
  let xEmbedding = positionRows[xRow * positionRowStride + row];
  let yEmbedding = positionRows[yRow * positionRowStride + row];
  outputValues[linear] = sum + xEmbedding + yEmbedding;
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Vision patch embedding shader failed validation: ${compilationMessages.join("; ")}`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "gemma4-vision-patch-embedding",
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: "gemma4-vision-patch-embedding",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: patchBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: positionRowsBuffer } },
        { binding: 3, resource: { buffer: patchPositionsBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: dimsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: "gemma4-vision-patch-embedding" });
    const pass = encoder.beginComputePass({ label: "gemma4-vision-patch-embedding" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((input.positions.length * input.outputRows) / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(input.positions.length * input.outputRows);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    patchBuffer.destroy();
    weightBuffer.destroy();
    positionRowsBuffer.destroy();
    patchPositionsBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}
