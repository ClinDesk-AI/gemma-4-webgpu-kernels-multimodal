import { alignedBufferSize, shaderCompilationMessages, writeGpuBuffer } from "../gpu/runtime.js";
import { loadSafetensorsTensorData, scalarF32FromTensor } from "../io/safetensors.js";
import { GPU_BUFFER_USAGE_COPY_DST, GPU_BUFFER_USAGE_COPY_SRC, GPU_BUFFER_USAGE_MAP_READ, GPU_BUFFER_USAGE_STORAGE, GPU_BUFFER_USAGE_UNIFORM, GPU_MAP_MODE_READ } from "../model.js";
import { Gemma4MediaKernelProbe, PackedQatLinearCpuInput, PackedQatLinearGpuInput, PackedQatProjectionData, QatI8LinearGpuInput, QatI8LinearInput, QatProjectionTensorNames, SafetensorsHeader, SafetensorsTensorData } from "../types.js";

export function runF32LinearRowsCpu(
  input: Float32Array,
  weights: Float32Array,
  rows: number,
  inputDim: number,
  outputDim: number,
): Float32Array {
  if (input.length !== rows * inputDim) {
    throw new Error("F32 linear input length does not match dimensions.");
  }
  if (weights.length !== inputDim * outputDim) {
    throw new Error("F32 linear weight length does not match dimensions.");
  }
  const output = new Float32Array(rows * outputDim);
  for (let row = 0; row < rows; row += 1) {
    const inputOffset = row * inputDim;
    const outputOffset = row * outputDim;
    for (let out = 0; out < outputDim; out += 1) {
      const weightOffset = out * inputDim;
      let sum = 0;
      for (let col = 0; col < inputDim; col += 1) {
        sum += input[inputOffset + col] * weights[weightOffset + col];
      }
      output[outputOffset + out] = sum;
    }
  }
  return output;
}

export async function runF32LinearRowsWebGpu(
  device: GPUDevice,
  input: {
    input: Float32Array;
    weights: Float32Array;
    rows: number;
    inputDim: number;
    outputDim: number;
    label: string;
  },
): Promise<Float32Array> {
  const outputBytes = input.rows * input.outputDim * 4;
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
      input.inputDim,
      input.outputDim,
      0,
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
  let linear = id.x;
  let rows = dims.x;
  let inputDim = dims.y;
  let outputDim = dims.z;
  if (linear >= rows * outputDim) {
    return;
  }
  let row = linear / outputDim;
  let out = linear % outputDim;
  let inputOffset = row * inputDim;
  let weightOffset = out * inputDim;
  var sum = 0.0;
  for (var col = 0u; col < inputDim; col = col + 1u) {
    sum = sum + inputValues[inputOffset + col] * weights[weightOffset + col];
  }
  outputValues[linear] = sum;
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`F32 linear shader failed validation: ${compilationMessages.join("; ")}`);
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
    pass.dispatchWorkgroups(Math.ceil((input.rows * input.outputDim) / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(input.rows * input.outputDim);
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

export async function loadPackedQatProjection<TProjection extends string>(input: {
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  names: QatProjectionTensorNames;
  projection: TProjection;
  inputDim: number;
  outputDim: number;
  signal?: AbortSignal;
}): Promise<PackedQatProjectionData<TProjection>> {
  const [weightData, weightScaleData, inputScaleData, outputScaleData] = await Promise.all([
    loadSafetensorsTensorData(input.urls.safetensors, input.header, input.names.weight, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, input.names.weightScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, input.names.inputScale, input.signal),
    loadSafetensorsTensorData(input.urls.safetensors, input.header, input.names.outputScale, input.signal),
  ]);
  const bits = derivePackedQatBits(weightData, input.inputDim, input.outputDim, input.projection);
  if (weightData.dtype !== "U8" ||
    weightData.shape.length !== 2 ||
    weightData.shape[0] !== input.outputDim ||
    weightData.shape[1] !== input.inputDim * bits / 8
  ) {
    throw new Error(`Unexpected ${input.projection} packed weight tensor: ${weightData.dtype} [${weightData.shape.join(", ")}].`);
  }
  if (weightScaleData.dtype !== "F32" ||
    weightScaleData.shape.length !== 2 ||
    weightScaleData.shape[0] !== input.outputDim ||
    weightScaleData.shape[1] !== 1
  ) {
    throw new Error(`Unexpected ${input.projection} weight scale tensor: ${weightScaleData.dtype} [${weightScaleData.shape.join(", ")}].`);
  }
  return {
    projection: input.projection,
    weightData,
    weightScaleData,
    bits,
    inputActivationScale: scalarF32FromTensor(inputScaleData, input.names.inputScale),
    outputActivationScale: scalarF32FromTensor(outputScaleData, input.names.outputScale),
  };
}

export function derivePackedQatBits(
  weightData: SafetensorsTensorData,
  inputDim: number,
  outputDim: number,
  label: string,
): number {
  const bits = weightData.bytes.byteLength * 8 / (inputDim * outputDim);
  if (!Number.isInteger(bits) || (bits !== 2 && bits !== 4 && bits !== 8)) {
    throw new Error(`${label}: derived unsupported ${bits}-bit QAT weights from ${weightData.bytes.byteLength} bytes.`);
  }
  if ((inputDim * bits) % 8 !== 0 || (inputDim * bits) % 32 !== 0) {
    throw new Error(`${label}: packed QAT row is not byte and u32 aligned.`);
  }
  return bits;
}

export function runPackedQatLinearCpu(input: PackedQatLinearCpuInput): Float32Array {
  if (input.input.length !== input.rows * input.inputDim) {
    throw new Error("Packed QAT input length does not match dimensions.");
  }
  const expectedBytes = input.outputDim * input.inputDim * input.bits / 8;
  if (input.weights.byteLength !== expectedBytes) {
    throw new Error(`Packed QAT weight length ${input.weights.byteLength} does not match ${expectedBytes}.`);
  }
  if (input.weightScales.length !== input.outputDim) {
    throw new Error("Packed QAT weight scale length does not match output dimension.");
  }
  const output = new Float32Array(input.rows * input.outputDim);
  const zeroPoint = 2 ** (input.bits - 1);
  for (let row = 0; row < input.rows; row += 1) {
    const inputOffset = row * input.inputDim;
    const outputOffset = row * input.outputDim;
    for (let out = 0; out < input.outputDim; out += 1) {
      let sumQA = 0;
      let sumA = 0;
      for (let col = 0; col < input.inputDim; col += 1) {
        const activation = srqScalar(input.input[inputOffset + col], input.inputActivationScale);
        const code = packedQatWeightCode(input.weights, out, col, input.inputDim, input.bits);
        sumQA = Math.fround(sumQA + Math.fround(code * activation));
        sumA = Math.fround(sumA + activation);
      }
      output[outputOffset + out] = srqScalar(
        Math.fround(input.weightScales[out] * Math.fround(sumQA - Math.fround(zeroPoint * sumA))),
        input.outputActivationScale,
      );
    }
  }
  return output;
}

export function packedQatWeightCode(
  weights: Uint8Array,
  out: number,
  col: number,
  inputDim: number,
  bits: number,
): number {
  const valuesPerByte = 8 / bits;
  const bytesPerRow = inputDim * bits / 8;
  const byteIndex = out * bytesPerRow + Math.floor(col / valuesPerByte);
  const lane = col % valuesPerByte;
  return (weights[byteIndex] >> (lane * bits)) & ((1 << bits) - 1);
}

export async function runPackedQatLinearWebGpu(
  device: GPUDevice,
  input: PackedQatLinearGpuInput,
): Promise<Float32Array> {
  if (input.weights.byteLength % 4 !== 0) {
    throw new Error("Packed QAT WebGPU weights must be 4-byte aligned.");
  }
  const outputBytes = input.rows * input.outputDim * 4;
  const inputBuffer = device.createBuffer({
    label: `${input.label}-input`,
    size: alignedBufferSize(input.input.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightBuffer = device.createBuffer({
    label: `${input.label}-weights-packed`,
    size: alignedBufferSize(input.weights.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightScaleBuffer = device.createBuffer({
    label: `${input.label}-weight-scales`,
    size: alignedBufferSize(input.weightScales.byteLength),
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
    const wordsPerRow = input.inputDim * input.bits / 32;
    const mask = (1 << input.bits) - 1;
    const zeroPoint = 2 ** (input.bits - 1);
    writeGpuBuffer(device, inputBuffer, 0, input.input);
    writeGpuBuffer(device, weightBuffer, 0, input.weights);
    writeGpuBuffer(device, weightScaleBuffer, 0, input.weightScales);
    writeGpuBuffer(device, paramsBuffer, 0, new Uint32Array([
      input.rows,
      input.inputDim,
      input.outputDim,
      input.bits,
      wordsPerRow,
      mask,
      input.rows * input.outputDim,
      0,
    ]));
    writeGpuBuffer(device, paramsBuffer, 32, new Float32Array([
      input.inputActivationScale,
      input.outputActivationScale,
      zeroPoint,
      0,
    ]));

    const module = device.createShaderModule({
      label: input.label,
      code: `
struct Params {
  rows: u32,
  inputDim: u32,
  outputDim: u32,
  bits: u32,
  wordsPerRow: u32,
  mask: u32,
  totalOutputValues: u32,
  _pad: u32,
  inputActivationScale: f32,
  outputActivationScale: f32,
  zeroPoint: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> packedWeights: array<u32>;
@group(0) @binding(2) var<storage, read> weightScales: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) {
    return value;
  }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

fn packedCode(out: u32, col: u32) -> f32 {
  let valuesPerWord = 32u / params.bits;
  let wordIndex = out * params.wordsPerRow + col / valuesPerWord;
  let shift = (col % valuesPerWord) * params.bits;
  return f32((packedWeights[wordIndex] >> shift) & params.mask);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputIndex = id.x;
  if (outputIndex >= params.totalOutputValues) {
    return;
  }
  let row = outputIndex / params.outputDim;
  let out = outputIndex % params.outputDim;
  let inputOffset = row * params.inputDim;

  var sumQA = 0.0;
  var sumA = 0.0;
  for (var col = 0u; col < params.inputDim; col = col + 1u) {
    let activation = srq(inputValues[inputOffset + col], params.inputActivationScale);
    sumQA = sumQA + packedCode(out, col) * activation;
    sumA = sumA + activation;
  }
  outputValues[outputIndex] = srq(weightScales[out] * (sumQA - params.zeroPoint * sumA), params.outputActivationScale);
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Packed QAT linear shader failed validation: ${compilationMessages.join("; ")}`);
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
        { binding: 2, resource: { buffer: weightScaleBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: input.label });
    const pass = encoder.beginComputePass({ label: input.label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((input.rows * input.outputDim) / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(input.rows * input.outputDim);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    inputBuffer.destroy();
    weightBuffer.destroy();
    weightScaleBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export function runProjectionCpu(
  input: Float32Array,
  weights: Float32Array,
  inputDim: number,
  outputDim: number,
): Float32Array {
  if (input.length !== inputDim) {
    throw new Error(`Projection input length ${input.length} does not match ${inputDim}.`);
  }
  if (weights.length !== inputDim * outputDim) {
    throw new Error(`Projection weight length ${weights.length} does not match ${outputDim} x ${inputDim}.`);
  }
  const output = new Float32Array(outputDim);
  for (let row = 0; row < outputDim; row += 1) {
    let sum = 0;
    const rowOffset = row * inputDim;
    for (let col = 0; col < inputDim; col += 1) {
      sum += input[col] * weights[rowOffset + col];
    }
    output[row] = sum;
  }
  return output;
}

export function runQatI8LinearCpu(input: QatI8LinearInput): Float32Array {
  if (input.input.length !== input.rows * input.inputDim) {
    throw new Error("QAT I8 input length does not match dimensions.");
  }
  if (input.weights.length !== input.outputDim * input.inputDim) {
    throw new Error("QAT I8 weight length does not match dimensions.");
  }
  if (input.weightScales.length !== input.outputDim) {
    throw new Error("QAT I8 weight scale length does not match output dimension.");
  }
  const output = new Float32Array(input.rows * input.outputDim);
  for (let row = 0; row < input.rows; row += 1) {
    const inputOffset = row * input.inputDim;
    const outputOffset = row * input.outputDim;
    for (let out = 0; out < input.outputDim; out += 1) {
      const weightOffset = out * input.inputDim;
      let sum = 0;
      for (let col = 0; col < input.inputDim; col += 1) {
        const product = srqScalar(input.input[inputOffset + col], input.inputActivationScale) *
          input.weights[weightOffset + col];
        sum = Math.fround(sum + Math.fround(product));
      }
      output[outputOffset + out] = srqScalar(
        Math.fround(sum * input.weightScales[out]),
        input.outputActivationScale,
      );
    }
  }
  return output;
}

export function srqScalar(value: number, scale: number): number {
  if (scale === 0) return value;
  return Math.max(-128, Math.min(127, Math.round(value / scale))) * scale;
}

export async function runProjectionWebGpu(
  device: GPUDevice,
  input: Float32Array,
  weights: Float32Array,
  inputDim: number,
  outputDim: number,
): Promise<Float32Array> {
  const outputBytes = outputDim * 4;
  const weightBuffer = device.createBuffer({
    label: "gemma4-media-projection-weight",
    size: alignedBufferSize(weights.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const inputBuffer = device.createBuffer({
    label: "gemma4-media-projection-input",
    size: alignedBufferSize(input.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: "gemma4-media-projection-output",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: "gemma4-media-projection-dims",
    size: 16,
    usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: "gemma4-media-projection-readback",
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
  });

  try {
    writeGpuBuffer(device, weightBuffer, 0, weights);
    writeGpuBuffer(device, inputBuffer, 0, input);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([inputDim, outputDim, 0, 0]));

    const module = device.createShaderModule({
      label: "gemma4-media-projection",
      code: `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> inputValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let inputDimLocal = dims.x;
  let outputDimLocal = dims.y;
  if (row >= outputDimLocal) {
    return;
  }
  var sum = 0.0;
  let rowOffset = row * inputDimLocal;
  for (var col = 0u; col < inputDimLocal; col = col + 1u) {
    sum = sum + inputValues[col] * weights[rowOffset + col];
  }
  outputValues[row] = sum;
}
`,
    });
    const pipeline = await device.createComputePipelineAsync({
      label: "gemma4-media-projection",
      layout: "auto",
      compute: {
        module,
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      label: "gemma4-media-projection",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: weightBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: "gemma4-media-projection" });
    const pass = encoder.beginComputePass({ label: "gemma4-media-projection" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outputDim / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(outputDim);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    weightBuffer.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}

export async function runQatI8LinearWebGpu(
  device: GPUDevice,
  input: QatI8LinearGpuInput,
): Promise<Float32Array> {
  if (input.weights.byteLength % 4 !== 0) {
    throw new Error("QAT I8 WebGPU weights must be 4-byte aligned.");
  }
  const outputBytes = input.rows * input.outputDim * 4;
  const inputBuffer = device.createBuffer({
    label: `${input.label}-input`,
    size: alignedBufferSize(input.input.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightBuffer = device.createBuffer({
    label: `${input.label}-weights-i8`,
    size: alignedBufferSize(input.weights.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightScaleBuffer = device.createBuffer({
    label: `${input.label}-weight-scales`,
    size: alignedBufferSize(input.weightScales.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: `${input.label}-output`,
    size: alignedBufferSize(outputBytes),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
  });
  const dimsBuffer = device.createBuffer({
    label: `${input.label}-dims`,
    size: 32,
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
    writeGpuBuffer(device, weightScaleBuffer, 0, input.weightScales);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([
      input.rows,
      input.inputDim,
      input.outputDim,
      0,
    ]));
    writeGpuBuffer(device, dimsBuffer, 16, new Float32Array([
      input.inputActivationScale,
      input.outputActivationScale,
      0,
      0,
    ]));

    const module = device.createShaderModule({
      label: input.label,
      code: `
struct Params {
  rows: u32,
  inputDim: u32,
  outputDim: u32,
  _pad: u32,
  inputActivationScale: f32,
  outputActivationScale: f32,
  _pad2: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> packedWeights: array<u32>;
@group(0) @binding(2) var<storage, read> weightScales: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) {
    return value;
  }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

fn i8Lane(word: u32, lane: u32) -> f32 {
  let byteValue = (word >> (lane * 8u)) & 255u;
  let signedValue = select(i32(byteValue), i32(byteValue) - 256, byteValue >= 128u);
  return f32(signedValue);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputIndex = id.x;
  let totalOutputValues = params.rows * params.outputDim;
  if (outputIndex >= totalOutputValues) {
    return;
  }
  let row = outputIndex / params.outputDim;
  let out = outputIndex % params.outputDim;
  let inputOffset = row * params.inputDim;
  let weightOffset = out * params.inputDim;
  var sum = 0.0;
  for (var col = 0u; col < params.inputDim; col = col + 1u) {
    let byteIndex = weightOffset + col;
    let packedWord = packedWeights[byteIndex / 4u];
    let weight = i8Lane(packedWord, byteIndex % 4u);
    let activation = srq(inputValues[inputOffset + col], params.inputActivationScale);
    sum = sum + activation * weight;
  }
  outputValues[outputIndex] = srq(sum * weightScales[out], params.outputActivationScale);
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`QAT I8 linear shader failed validation: ${compilationMessages.join("; ")}`);
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
        { binding: 2, resource: { buffer: weightScaleBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: dimsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: input.label });
    const pass = encoder.beginComputePass({ label: input.label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((input.rows * input.outputDim) / 64));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
    const mapped = readbackBuffer.getMappedRange(0, outputBytes);
    const output = new Float32Array(input.rows * input.outputDim);
    output.set(new Float32Array(mapped.slice(0)));
    readbackBuffer.unmap();
    return output;
  } finally {
    inputBuffer.destroy();
    weightBuffer.destroy();
    weightScaleBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}
