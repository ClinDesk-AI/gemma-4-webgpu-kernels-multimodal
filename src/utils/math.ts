import { Gemma4VisionFeedForwardKernelProbe, SafetensorsTensorData } from "../types.js";

export function checksumFloats(values: Float32Array): number {
  let checksum = 0;
  const modulo = 1_000_000_007;
  const limit = Math.min(values.length, 4096);
  for (let index = 0; index < limit; index += 1) {
    checksum = (checksum + Math.round(values[index] * 1_000_000) * (index + 1)) % modulo;
    if (checksum < 0) checksum += modulo;
  }
  return checksum;
}

export function checksumBytes(bytes: Uint8Array): number {
  let checksum = 0;
  const modulo = 1_000_000_007;
  for (let index = 0; index < bytes.length; index += 1) {
    checksum = (checksum + bytes[index] * (index + 1)) % modulo;
  }
  return checksum;
}

export function allFinite(values: Float32Array): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) return false;
  }
  return true;
}

export function tensorSummaryForProbe(
  tensor: SafetensorsTensorData,
): NonNullable<Gemma4VisionFeedForwardKernelProbe["tensors"]["preFeedForwardLayerNorm"]> {
  return {
    name: tensor.name,
    dtype: tensor.dtype,
    shape: tensor.shape,
    dataBytes: tensor.dataBytes,
    fromCache: tensor.fromCache,
    checksum: checksumBytes(tensor.bytes),
  };
}

export function deterministicProjectionInput(inputDim: number): Float32Array {
  const input = new Float32Array(inputDim);
  for (let index = 0; index < input.length; index += 1) {
    input[index] = Math.sin(index * 0.017) * 0.125 + Math.cos(index * 0.031) * 0.0625;
  }
  return input;
}

export function deterministicPatchValues(patches: number, patchPixels: number): Float32Array {
  const values = new Float32Array(patches * patchPixels);
  for (let patch = 0; patch < patches; patch += 1) {
    const patchOffset = patch * patchPixels;
    for (let index = 0; index < patchPixels; index += 1) {
      values[patchOffset + index] = ((index * 17 + patch * 29) % 256) / 255;
    }
  }
  return values;
}

export function deterministicVisionHiddenStates(rows: number, hiddenSize: number): Float32Array {
  return deterministicMultimodalHiddenStates(rows, hiddenSize);
}

export function deterministicMultimodalHiddenStates(rows: number, hiddenSize: number): Float32Array {
  const values = new Float32Array(rows * hiddenSize);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * hiddenSize;
    for (let index = 0; index < hiddenSize; index += 1) {
      values[offset + index] =
        Math.sin((index + 1) * 0.013 + row * 0.17) * 1.25 +
        Math.cos((index + 1) * 0.029 + row * 0.11) * 0.375;
    }
  }
  return values;
}

export function visionRopePositions(rows: number): Array<{ x: number; y: number }> {
  return Array.from({ length: rows }, (_, index) => ({
    x: index % 2,
    y: Math.floor(index / 2),
  }));
}

export function onesFloat32(length: number): Float32Array {
  const values = new Float32Array(length);
  values.fill(1);
  return values;
}

export function maxAbsDifference(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Cannot compare arrays of length ${a.length} and ${b.length}.`);
  }
  let max = 0;
  for (let index = 0; index < a.length; index += 1) {
    max = Math.max(max, Math.abs(a[index] - b[index]));
  }
  return max;
}

export function maxAbsDifferenceStats(
  a: Float32Array,
  b: Float32Array,
  tolerance: number,
): { maxAbsDiff: number; index: number; countAboveTolerance: number; cpuValue: number | null; gpuValue: number | null } {
  if (a.length !== b.length) {
    throw new Error(`Cannot compare arrays of length ${a.length} and ${b.length}.`);
  }
  let maxAbsDiff = 0;
  let index = -1;
  let countAboveTolerance = 0;
  for (let valueIndex = 0; valueIndex < a.length; valueIndex += 1) {
    const diff = Math.abs(a[valueIndex] - b[valueIndex]);
    if (diff > tolerance) countAboveTolerance += 1;
    if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
      index = valueIndex;
    }
  }
  return {
    maxAbsDiff,
    index,
    countAboveTolerance,
    cpuValue: index >= 0 ? a[index] : null,
    gpuValue: index >= 0 ? b[index] : null,
  };
}

export function roundedSample(values: Float32Array): number[] {
  return Array.from(values.slice(0, 8), (value) => Math.round(value * 1_000_000) / 1_000_000);
}