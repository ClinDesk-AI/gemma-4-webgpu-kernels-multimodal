import { Gemma4ImagePreprocessOptions, Gemma4ImagePreprocessProbe, Gemma4ImagePreprocessResult, Gemma4ImageTokenLayout } from "../types.js";
import { checksumFloats } from "../utils/math.js";

export function computeGemma4ImageTokenLayout(input: {
  height: number;
  width: number;
  patchSize: number;
  maxSoftTokens: number;
  poolingKernelSize: number;
}): Gemma4ImageTokenLayout {
  const maxPatches = input.maxSoftTokens * input.poolingKernelSize ** 2;
  const target = aspectRatioPreservingSize({
    height: input.height,
    width: input.width,
    patchSize: input.patchSize,
    maxPatches,
    poolingKernelSize: input.poolingKernelSize,
  });
  const patchGrid = {
    height: target.height / input.patchSize,
    width: target.width / input.patchSize,
  };
  return {
    input: { height: input.height, width: input.width },
    target,
    patchGrid,
    maxPatches,
    softTokens: Math.floor((patchGrid.height * patchGrid.width) / input.poolingKernelSize ** 2),
  };
}

export async function preprocessGemma4ImageBlob(
  blob: Blob,
  options: Gemma4ImagePreprocessOptions = {},
): Promise<Gemma4ImagePreprocessResult> {
  const bitmap = await createImageBitmap(blob);
  try {
    return preprocessGemma4ImageSource(bitmap, bitmap.width, bitmap.height, options);
  } finally {
    bitmap.close();
  }
}

export async function probeGemma4ImagePreprocessing(): Promise<Gemma4ImagePreprocessProbe> {
  try {
    const width = 320;
    const height = 240;
    const source = createCanvas(width, height);
    const context = canvasContext(source);
    const imageData = context.createImageData(width, height);
    for (let offset = 0; offset < imageData.data.length; offset += 4) {
      const pixel = offset / 4;
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      imageData.data[offset] = x % 256;
      imageData.data[offset + 1] = y % 256;
      imageData.data[offset + 2] = (x + y) % 256;
      imageData.data[offset + 3] = 255;
    }
    context.putImageData(imageData, 0, 0);

    const result = await preprocessGemma4ImageSource(
      source as CanvasImageSource,
      width,
      height,
    );
    return {
      ok: true,
      source: { width, height },
      layout: result.layout,
      pixelValueLength: result.pixelValues.length,
      imagePositionIdLength: result.imagePositionIds.length,
      numSoftTokens: result.numSoftTokens,
      maxPatches: result.maxPatches,
      patchPixels: result.patchPixels,
      firstPositionIds: Array.from(result.imagePositionIds.slice(0, 10)),
      paddedPositionTail: Array.from(result.imagePositionIds.slice(-10)),
      pixelChecksum: checksumFloats(result.pixelValues),
    };
  } catch (error) {
    return {
      ok: false,
      source: { width: 0, height: 0 },
      layout: computeGemma4ImageTokenLayout({
        height: 1,
        width: 1,
        patchSize: 16,
        maxSoftTokens: 280,
        poolingKernelSize: 3,
      }),
      pixelValueLength: 0,
      imagePositionIdLength: 0,
      numSoftTokens: 0,
      maxPatches: 0,
      patchPixels: 0,
      firstPositionIds: [],
      paddedPositionTail: [],
      pixelChecksum: 0,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function preprocessGemma4ImageSource(
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  options: Gemma4ImagePreprocessOptions = {},
): Promise<Gemma4ImagePreprocessResult> {
  const patchSize = options.patchSize ?? 16;
  const maxSoftTokens = options.maxSoftTokens ?? 280;
  const poolingKernelSize = options.poolingKernelSize ?? 3;
  const rescaleFactor = options.rescaleFactor ?? (1 / 255);
  const layout = computeGemma4ImageTokenLayout({
    height: sourceHeight,
    width: sourceWidth,
    patchSize,
    maxSoftTokens,
    poolingKernelSize,
  });
  const target = createCanvas(layout.target.width, layout.target.height);
  const context = canvasContext(target);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, layout.target.width, layout.target.height);
  const imageData = context.getImageData(0, 0, layout.target.width, layout.target.height);

  const patchPixels = patchSize * patchSize * 3;
  const pixelValues = new Float32Array(layout.maxPatches * patchPixels);
  const imagePositionIds = new Int32Array(layout.maxPatches * 2);
  imagePositionIds.fill(-1);

  let patchIndex = 0;
  for (let patchY = 0; patchY < layout.patchGrid.height; patchY += 1) {
    for (let patchX = 0; patchX < layout.patchGrid.width; patchX += 1) {
      imagePositionIds[patchIndex * 2] = patchX;
      imagePositionIds[patchIndex * 2 + 1] = patchY;
      const patchOffset = patchIndex * patchPixels;
      for (let dy = 0; dy < patchSize; dy += 1) {
        const sourceY = patchY * patchSize + dy;
        for (let dx = 0; dx < patchSize; dx += 1) {
          const sourceX = patchX * patchSize + dx;
          const rgbaOffset = (sourceY * layout.target.width + sourceX) * 4;
          const patchPixelOffset = patchOffset + (dy * patchSize + dx) * 3;
          pixelValues[patchPixelOffset] = imageData.data[rgbaOffset] * rescaleFactor;
          pixelValues[patchPixelOffset + 1] = imageData.data[rgbaOffset + 1] * rescaleFactor;
          pixelValues[patchPixelOffset + 2] = imageData.data[rgbaOffset + 2] * rescaleFactor;
        }
      }
      patchIndex += 1;
    }
  }

  return {
    layout,
    pixelValues,
    imagePositionIds,
    numSoftTokens: layout.softTokens,
    maxPatches: layout.maxPatches,
    patchPixels,
  };
}

export function createDeterministicImageCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  const source = createCanvas(width, height);
  const context = canvasContext(source);
  const imageData = context.createImageData(width, height);
  for (let offset = 0; offset < imageData.data.length; offset += 4) {
    const pixel = offset / 4;
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    imageData.data[offset] = x % 256;
    imageData.data[offset + 1] = y % 256;
    imageData.data[offset + 2] = (x + y) % 256;
    imageData.data[offset + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return source;
}

export function validImagePatchInput(input: Gemma4ImagePreprocessResult): {
  patchValues: Float32Array;
  positions: Array<{ x: number; y: number }>;
} {
  const positions: Array<{ x: number; y: number }> = [];
  for (let patchIndex = 0; patchIndex < input.maxPatches; patchIndex += 1) {
    const x = input.imagePositionIds[patchIndex * 2];
    const y = input.imagePositionIds[patchIndex * 2 + 1];
    if (x < 0 || y < 0) break;
    positions.push({ x, y });
  }
  if (positions.length === 0) {
    throw new Error("Image preprocessing produced zero valid patches.");
  }
  const patchValues = input.pixelValues.slice(0, positions.length * input.patchPixels);
  return { patchValues, positions };
}

export function createCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function canvasContext(canvas: OffscreenCanvas | HTMLCanvasElement): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D {
  const context = canvas instanceof HTMLCanvasElement
    ? canvas.getContext("2d", { willReadFrequently: true })
    : canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not create a 2D canvas context.");
  return context;
}

export function aspectRatioPreservingSize(input: {
  height: number;
  width: number;
  patchSize: number;
  maxPatches: number;
  poolingKernelSize: number;
}): { height: number; width: number } {
  if (input.height <= 0 || input.width <= 0) {
    throw new Error("Image height and width must be positive.");
  }
  const totalPixels = input.height * input.width;
  const targetPixels = input.maxPatches * input.patchSize ** 2;
  const factor = Math.sqrt(targetPixels / totalPixels);
  const idealHeight = factor * input.height;
  const idealWidth = factor * input.width;
  const sideMultiple = input.poolingKernelSize * input.patchSize;

  let targetHeight = Math.floor(idealHeight / sideMultiple) * sideMultiple;
  let targetWidth = Math.floor(idealWidth / sideMultiple) * sideMultiple;
  if (targetHeight === 0 && targetWidth === 0) {
    throw new Error(`Image resize rounded to 0 x 0 for side multiple ${sideMultiple}.`);
  }

  const maxSideLength = Math.floor(input.maxPatches / input.poolingKernelSize ** 2) * sideMultiple;
  if (targetHeight === 0) {
    targetHeight = sideMultiple;
    targetWidth = Math.min(Math.floor(input.width / input.height) * sideMultiple, maxSideLength);
  } else if (targetWidth === 0) {
    targetWidth = sideMultiple;
    targetHeight = Math.min(Math.floor(input.height / input.width) * sideMultiple, maxSideLength);
  }
  if (targetHeight * targetWidth > targetPixels) {
    throw new Error(`Image resize exceeds ${input.maxPatches} patches.`);
  }
  return { height: targetHeight, width: targetWidth };
}