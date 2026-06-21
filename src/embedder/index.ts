import { alignedBufferSize, shaderCompilationMessages, writeGpuBuffer } from "../gpu/runtime.js";
import { float32FromBytes, loadSafetensorsTensorData } from "../io/safetensors.js";
import { GPU_BUFFER_USAGE_COPY_DST, GPU_BUFFER_USAGE_COPY_SRC, GPU_BUFFER_USAGE_MAP_READ, GPU_BUFFER_USAGE_STORAGE, GPU_BUFFER_USAGE_UNIFORM, GPU_MAP_MODE_READ } from "../model.js";
import { Gemma4MediaKernelProbe, MultimodalEmbedderGpuInput, SafetensorsHeader, SafetensorsTensorData } from "../types.js";
import { maxAbsDifference } from "../utils/math.js";

export async function computeMultimodalEmbedder(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  kind: "vision" | "audio";
  hiddenStates: Float32Array;
  gpuHiddenStates?: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<{
  projectionData: SafetensorsTensorData;
  cpuOutput: Float32Array;
  gpuOutput: Float32Array;
  maxAbsDiff: number;
}> {
  const projectionName = input.kind === "vision"
    ? "model.embed_vision.embedding_projection.weight"
    : "model.embed_audio.embedding_projection.weight";
  const projectionData = await loadSafetensorsTensorData(
    input.urls.safetensors,
    input.header,
    projectionName,
    input.signal,
  );
  if (projectionData.dtype !== "F32" ||
    projectionData.shape.length !== 2 ||
    projectionData.shape[0] !== input.outputDim ||
    projectionData.shape[1] !== input.inputDim
  ) {
    throw new Error(
      `Unexpected ${input.kind} embedder projection tensor: ${projectionData.dtype} [${projectionData.shape.join(", ")}].`,
    );
  }

  const weights = float32FromBytes(projectionData.bytes);
  const cpuOutput = runMultimodalEmbedderCpu(
    input.hiddenStates,
    weights,
    input.rows,
    input.inputDim,
    input.outputDim,
    input.epsilon,
  );
  const gpuOutput = await runMultimodalEmbedderWebGpu(input.device, {
    hiddenStates: input.gpuHiddenStates ?? input.hiddenStates,
    weights,
    rows: input.rows,
    inputDim: input.inputDim,
    outputDim: input.outputDim,
    epsilon: input.epsilon,
    label: `gemma4-${input.kind}-embedder`,
  });
  return {
    projectionData,
    cpuOutput,
    gpuOutput,
    maxAbsDiff: maxAbsDifference(cpuOutput, gpuOutput),
  };
}

export async function computeMultimodalEmbedderWebGpuOnly(input: {
  device: GPUDevice;
  urls: Gemma4MediaKernelProbe["urls"];
  header: SafetensorsHeader;
  kind: "vision" | "audio";
  hiddenStates: Float32Array;
  rows: number;
  inputDim: number;
  outputDim: number;
  epsilon: number;
  signal?: AbortSignal;
}): Promise<Float32Array> {
  const projectionName = input.kind === "vision"
    ? "model.embed_vision.embedding_projection.weight"
    : "model.embed_audio.embedding_projection.weight";
  const projectionData = await loadSafetensorsTensorData(
    input.urls.safetensors,
    input.header,
    projectionName,
    input.signal,
  );
  if (projectionData.dtype !== "F32" ||
    projectionData.shape.length !== 2 ||
    projectionData.shape[0] !== input.outputDim ||
    projectionData.shape[1] !== input.inputDim
  ) {
    throw new Error(
      `Unexpected ${input.kind} embedder projection tensor: ${projectionData.dtype} [${projectionData.shape.join(", ")}].`,
    );
  }
  return runMultimodalEmbedderWebGpu(input.device, {
    hiddenStates: input.hiddenStates,
    weights: float32FromBytes(projectionData.bytes),
    rows: input.rows,
    inputDim: input.inputDim,
    outputDim: input.outputDim,
    epsilon: input.epsilon,
    label: `gemma4-${input.kind}-embedder`,
  });
}

export function runMultimodalEmbedderCpu(
  hiddenStates: Float32Array,
  weights: Float32Array,
  rows: number,
  inputDim: number,
  outputDim: number,
  epsilon: number,
): Float32Array {
  if (hiddenStates.length !== rows * inputDim) {
    throw new Error("Multimodal embedder hidden-state length does not match dimensions.");
  }
  if (weights.length !== inputDim * outputDim) {
    throw new Error("Multimodal embedder projection weight length does not match dimensions.");
  }
  const output = new Float32Array(rows * outputDim);
  for (let row = 0; row < rows; row += 1) {
    const inputOffset = row * inputDim;
    const outputOffset = row * outputDim;
    let meanSquare = 0;
    for (let index = 0; index < inputDim; index += 1) {
      const value = hiddenStates[inputOffset + index];
      meanSquare += value * value;
    }
    const invRms = (meanSquare / inputDim + epsilon) ** -0.5;
    for (let out = 0; out < outputDim; out += 1) {
      const weightOffset = out * inputDim;
      let sum = 0;
      for (let col = 0; col < inputDim; col += 1) {
        sum += hiddenStates[inputOffset + col] * invRms * weights[weightOffset + col];
      }
      output[outputOffset + out] = sum;
    }
  }
  return output;
}

export async function runMultimodalEmbedderWebGpu(
  device: GPUDevice,
  input: MultimodalEmbedderGpuInput,
): Promise<Float32Array> {
  const outputBytes = input.rows * input.outputDim * 4;
  const hiddenBuffer = device.createBuffer({
    label: `${input.label}-hidden`,
    size: alignedBufferSize(input.hiddenStates.byteLength),
    usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
  });
  const weightBuffer = device.createBuffer({
    label: `${input.label}-weight`,
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
    writeGpuBuffer(device, hiddenBuffer, 0, input.hiddenStates);
    writeGpuBuffer(device, weightBuffer, 0, input.weights);
    writeGpuBuffer(device, dimsBuffer, 0, new Uint32Array([
      input.rows,
      input.inputDim,
      input.outputDim,
      Math.round(input.epsilon * 1_000_000_000),
    ]));

    const module = device.createShaderModule({
      label: input.label,
      code: `
@group(0) @binding(0) var<storage, read> hiddenStates: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputIndex = id.x;
  let rows = dims.x;
  let inputDim = dims.y;
  let outputDim = dims.z;
  let epsilon = f32(dims.w) / 1000000000.0;
  let totalOutputValues = rows * outputDim;
  if (outputIndex >= totalOutputValues) {
    return;
  }
  let row = outputIndex / outputDim;
  let out = outputIndex % outputDim;
  let inputOffset = row * inputDim;
  var meanSquare = 0.0;
  for (var index = 0u; index < inputDim; index = index + 1u) {
    let value = hiddenStates[inputOffset + index];
    meanSquare = meanSquare + value * value;
  }
  let invRms = inverseSqrt(meanSquare / f32(inputDim) + epsilon);
  var sum = 0.0;
  let weightOffset = out * inputDim;
  for (var col = 0u; col < inputDim; col = col + 1u) {
    sum = sum + hiddenStates[inputOffset + col] * invRms * weights[weightOffset + col];
  }
  outputValues[outputIndex] = sum;
}
`,
    });
    const compilationMessages = await shaderCompilationMessages(module);
    if (compilationMessages.length > 0) {
      throw new Error(`Multimodal embedder shader failed validation: ${compilationMessages.join("; ")}`);
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
        { binding: 0, resource: { buffer: hiddenBuffer } },
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
    hiddenBuffer.destroy();
    weightBuffer.destroy();
    outputBuffer.destroy();
    dimsBuffer.destroy();
    readbackBuffer.destroy();
  }
}
