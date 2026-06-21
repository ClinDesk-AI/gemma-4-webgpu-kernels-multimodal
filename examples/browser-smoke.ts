import {
  computeGemma4AudioSoftTokenCount,
  computeGemma4ImageTokenLayout,
  preprocessGemma4AudioSamples,
  preprocessGemma4ImageBlob,
} from "@clindesk/gemma-4-webgpu-kernels-multimodal";

export async function inspectImageFile(file: Blob) {
  const preprocessed = await preprocessGemma4ImageBlob(file);
  const layout = computeGemma4ImageTokenLayout({
    width: preprocessed.width,
    height: preprocessed.height,
  });

  return {
    layout,
    pixelCount: preprocessed.pixels.length,
  };
}

export function inspectAudioSamples(samples: Float32Array, sampleRate: number) {
  const preprocessed = preprocessGemma4AudioSamples({ samples, sampleRate });
  const softTokens = computeGemma4AudioSoftTokenCount({
    sampleCount: preprocessed.samples.length,
    sampleRate: preprocessed.sampleRate,
  });

  return {
    softTokens,
    sampleRate: preprocessed.sampleRate,
    sampleCount: preprocessed.samples.length,
  };
}
