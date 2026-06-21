import { computeGemma4AudioSoftTokensFromMask } from "../audio/kernels.js";
import { Gemma4AudioPreprocessOptions, Gemma4AudioPreprocessResult } from "../types.js";

export function computeGemma4AudioSoftTokenCount(input: {
  samples: number;
  frameLength: number;
  hopLength: number;
  audioSeqLength: number;
}): number {
  const frameSizeForUnfold = input.frameLength + 1;
  const padLeft = Math.floor(input.frameLength / 2);
  const numMelFrames = Math.floor((input.samples + padLeft - frameSizeForUnfold) / input.hopLength) + 1;
  if (numMelFrames <= 0) return 0;

  let timeSteps = numMelFrames;
  for (let layer = 0; layer < 2; layer += 1) {
    timeSteps = Math.floor((timeSteps + 2 * 1 - 3) / 2) + 1;
  }
  return Math.min(timeSteps, input.audioSeqLength);
}

export function preprocessGemma4AudioSamples(
  rawSamples: Float32Array | number[],
  options: Gemma4AudioPreprocessOptions = {},
): Gemma4AudioPreprocessResult {
  const samplingRate = options.samplingRate ?? 16_000;
  const featureSize = options.featureSize ?? 128;
  const frameLength = options.frameLength ?? 320;
  const hopLength = options.hopLength ?? 160;
  const fftLength = options.fftLength ?? 512;
  const minFrequency = options.minFrequency ?? 0;
  const maxFrequency = options.maxFrequency ?? 8_000;
  const melFloor = options.melFloor ?? 0.001;
  const inputScaleFactor = options.inputScaleFactor ?? 1;
  const padToMultipleOf = options.padToMultipleOf === undefined ? 128 : options.padToMultipleOf;
  const maxSamples = options.maxSamples ?? 480_000;
  const audioSeqLength = options.audioSeqLength ?? 750;

  if (!isPowerOfTwo(fftLength)) throw new Error("Gemma 4 audio FFT length must be a power of two.");
  if (fftLength < frameLength) throw new Error("Gemma 4 audio FFT length must cover the frame length.");
  const sourceLength = Math.min(rawSamples.length, maxSamples);
  const paddedSourceLength = padToMultipleOf && padToMultipleOf > 0
    ? Math.ceil(sourceLength / padToMultipleOf) * padToMultipleOf
    : sourceLength;
  const padLeft = Math.floor(frameLength / 2);
  const waveform = new Float32Array(padLeft + paddedSourceLength);
  const attentionMask = new Uint8Array(waveform.length);
  for (let index = 0; index < sourceLength; index += 1) {
    waveform[padLeft + index] = rawSamples[index] * inputScaleFactor;
    attentionMask[padLeft + index] = 1;
  }

  const frameSizeForUnfold = frameLength + 1;
  const frames = waveform.length >= frameSizeForUnfold
    ? Math.floor((waveform.length - frameSizeForUnfold) / hopLength) + 1
    : 0;
  const inputFeatures = new Float32Array(frames * featureSize);
  const inputFeaturesMask = new Uint8Array(frames);
  if (frames === 0) {
    return {
      inputFeatures,
      inputFeaturesMask,
      frames,
      validFrames: 0,
      featureSize,
      samplingRate,
      fftLength,
      softTokens: 0,
    };
  }

  const window = periodicHannWindow(frameLength);
  const melFilters = createHtkMelFilterBank({
    numFrequencyBins: Math.floor(fftLength / 2) + 1,
    numMelFilters: featureSize,
    minFrequency,
    maxFrequency,
    samplingRate,
  });
  const fftReal = new Float32Array(fftLength);
  const fftImag = new Float32Array(fftLength);
  const magnitude = new Float32Array(Math.floor(fftLength / 2) + 1);
  let validFrames = 0;

  for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
    const frameStart = frameIndex * hopLength;
    fftReal.fill(0);
    fftImag.fill(0);
    for (let sampleIndex = 0; sampleIndex < frameLength; sampleIndex += 1) {
      fftReal[sampleIndex] = waveform[frameStart + sampleIndex] * window[sampleIndex];
    }
    realFftMagnitude(fftReal, fftImag, magnitude);

    const frameEndIndex = frameStart + frameSizeForUnfold - 1;
    const valid = attentionMask[frameEndIndex] === 1;
    if (valid) validFrames += 1;
    inputFeaturesMask[frameIndex] = valid ? 1 : 0;
    const featureOffset = frameIndex * featureSize;
    for (let melIndex = 0; melIndex < featureSize; melIndex += 1) {
      let melEnergy = 0;
      for (let binIndex = 0; binIndex < magnitude.length; binIndex += 1) {
        melEnergy += magnitude[binIndex] * melFilters[binIndex * featureSize + melIndex];
      }
      inputFeatures[featureOffset + melIndex] = valid ? Math.log(melEnergy + melFloor) : 0;
    }
  }

  return {
    inputFeatures,
    inputFeaturesMask,
    frames,
    validFrames,
    featureSize,
    samplingRate,
    fftLength,
    softTokens: computeGemma4AudioSoftTokensFromMask(inputFeaturesMask, audioSeqLength),
  };
}

export function periodicHannWindow(length: number): Float32Array {
  const window = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    window[index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / length);
  }
  return window;
}

export function createHtkMelFilterBank(input: {
  numFrequencyBins: number;
  numMelFilters: number;
  minFrequency: number;
  maxFrequency: number;
  samplingRate: number;
}): Float32Array {
  const filters = new Float32Array(input.numFrequencyBins * input.numMelFilters);
  const melMin = hertzToHtkMel(input.minFrequency);
  const melMax = hertzToHtkMel(input.maxFrequency);
  const filterFrequencies = new Float64Array(input.numMelFilters + 2);
  for (let index = 0; index < filterFrequencies.length; index += 1) {
    const mel = melMin + ((melMax - melMin) * index) / (filterFrequencies.length - 1);
    filterFrequencies[index] = htkMelToHertz(mel);
  }

  const fftBinWidth = input.samplingRate / ((input.numFrequencyBins - 1) * 2);
  for (let binIndex = 0; binIndex < input.numFrequencyBins; binIndex += 1) {
    const frequency = fftBinWidth * binIndex;
    for (let melIndex = 0; melIndex < input.numMelFilters; melIndex += 1) {
      const left = filterFrequencies[melIndex];
      const center = filterFrequencies[melIndex + 1];
      const right = filterFrequencies[melIndex + 2];
      const up = center === left ? 0 : (frequency - left) / (center - left);
      const down = right === center ? 0 : (right - frequency) / (right - center);
      filters[binIndex * input.numMelFilters + melIndex] = Math.max(0, Math.min(up, down));
    }
  }
  return filters;
}

export function realFftMagnitude(
  real: Float32Array,
  imag: Float32Array,
  magnitude: Float32Array,
): void {
  imag.fill(0);
  fftRadix2(real, imag);
  for (let index = 0; index < magnitude.length; index += 1) {
    magnitude[index] = Math.hypot(real[index], imag[index]);
  }
}

export function fftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  let j = 0;
  for (let i = 1; i < n; i += 1) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const realTmp = real[i];
      const imagTmp = imag[i];
      real[i] = real[j];
      imag[i] = imag[j];
      real[j] = realTmp;
      imag[j] = imagTmp;
    }
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const theta = -2 * Math.PI / size;
    const phaseStepReal = Math.cos(theta);
    const phaseStepImag = Math.sin(theta);
    for (let start = 0; start < n; start += size) {
      let phaseReal = 1;
      let phaseImag = 0;
      for (let offset = 0; offset < half; offset += 1) {
        const even = start + offset;
        const odd = even + half;
        const oddReal = real[odd] * phaseReal - imag[odd] * phaseImag;
        const oddImag = real[odd] * phaseImag + imag[odd] * phaseReal;
        real[odd] = real[even] - oddReal;
        imag[odd] = imag[even] - oddImag;
        real[even] += oddReal;
        imag[even] += oddImag;
        const nextPhaseReal = phaseReal * phaseStepReal - phaseImag * phaseStepImag;
        phaseImag = phaseReal * phaseStepImag + phaseImag * phaseStepReal;
        phaseReal = nextPhaseReal;
      }
    }
  }
}

export function hertzToHtkMel(frequency: number): number {
  return 2595 * Math.log10(1 + frequency / 700);
}

export function htkMelToHertz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

export function isPowerOfTwo(value: number): boolean {
  return value > 0 && (value & (value - 1)) === 0;
}