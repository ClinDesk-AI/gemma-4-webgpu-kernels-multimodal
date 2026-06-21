# gemma-4-webgpu-kernels-multimodal

Browser-native WebGPU media kernels for Gemma 4 E2B mobile-transformers image
and audio embeddings.

This package extracts the multimodal WebGPU kernel work from the ClinDesk browser
prototype into a standalone library. It is intended for browser apps that need to
turn image or audio inputs into decoder-ready embedding rows without a native
runtime, server process, ONNX runtime, or desktop bridge.

## Status

Experimental. The current package exposes the stable browser-facing API first,
with internal kernels split by concern. The `src/core/gemma4-media.ts` file is a
compatibility barrel, not the implementation.

## Structure

```text
src/
  index.ts                 public package API
  model.ts                 model IDs, cache names, tensor name helpers
  types.ts                 shared public and internal TypeScript contracts
  audio/
    index.ts               audio embedding entry point
    encoder.ts             audio tower orchestration
    kernels.ts             audio-specific WebGPU kernels
    probes.ts              audio verification probes
  embedder/
    index.ts               multimodal projection into decoder embedding space
    probes.ts              embedder verification probes
  gpu/
    runtime.ts             WebGPU helpers and typed buffer writes
  io/
    safetensors.ts         Hugging Face fetch, range, cache, tensor parsing
  kernels/
    qat.ts                 shared QAT and projection kernels
  preprocess/
    audio.ts               waveform to feature frames and masks
    image.ts               browser image decode, resize, patch extraction
  probes/
    media.ts               artifact, tensor, and projection smoke probes
  vision/
    index.ts               image embedding entry point
    encoder.ts             vision tower orchestration
    kernels.ts             vision-specific WebGPU kernels
    probes.ts              vision verification probes
```

## Install

```sh
pnpm add @clindesk/gemma-4-webgpu-kernels-multimodal
```

## Requirements

- A browser with WebGPU enabled.
- HTTPS or localhost.
- Access to `google/gemma-4-E2B-it-qat-mobile-transformers` on Hugging Face.
- User consent for downloading and caching model artifacts in browser storage.

This package does not bundle model weights.

## API

```ts
import {
  computeGemma4AudioEmbeddings,
  computeGemma4VisionImageEmbeddings,
  preprocessGemma4AudioSamples,
  preprocessGemma4ImageBlob,
} from "@clindesk/gemma-4-webgpu-kernels-multimodal";
```

### Images

```ts
const preprocessed = await preprocessGemma4ImageBlob(imageBlob);
const result = await computeGemma4VisionImageEmbeddings({
  pixels: preprocessed.pixels,
  width: preprocessed.width,
  height: preprocessed.height,
});

console.log(result.embeddings, result.softTokenCount);
```

### Audio

```ts
const preprocessed = preprocessGemma4AudioSamples({
  samples: monoFloat32Samples,
  sampleRate,
});

const result = await computeGemma4AudioEmbeddings({
  samples: preprocessed.samples,
  sampleRate: preprocessed.sampleRate,
});

console.log(result.embeddings, result.softTokenCount);
```

The output embedding rows are designed to be overlaid into the decoder input
embedding sequence at the corresponding media placeholder positions.

## Exports

- `computeGemma4VisionImageEmbeddings`
- `computeGemma4AudioEmbeddings`
- `preprocessGemma4ImageBlob`
- `preprocessGemma4AudioSamples`
- `computeGemma4ImageTokenLayout`
- `computeGemma4AudioSoftTokenCount`
- TypeScript result and preprocessing types

Kernel probes are available through the `./probes` export. Internal modules are
not treated as stable API yet.

## Performance Notes

- Product embedding paths use GPU-only tower execution. CPU/GPU comparison code
  is isolated in probe modules so bundlers can tree-shake it out of application
  builds.
- Tensor metadata and byte ranges are cached through the browser Cache API.
- The package avoids native runtimes and does not require ONNX, WASM sidecars, or
  server inference.
- `sideEffects: false` is set in `package.json`; apps should import from the root
  API unless they intentionally need the lower-level `./probes` checks.

## Storage

The library uses the browser Cache API for model metadata and tensor slices. Apps
should surface a clear onboarding step before the first model download and should
provide a way to clear local model data.

## Release

This repository uses pnpm for package management and publishing.

```sh
pnpm install --frozen-lockfile
pnpm build
pnpm probe:static
pnpm publish --access public
```

If the registry asks for a one-time password in a non-interactive environment,
publish with:

```sh
pnpm publish --access public --otp <code>
```

## Provenance

This package is a ClinDesk-authored extraction from the browser prototype. It was
developed against:

- `webml-community/gemma-4-webgpu-kernels`, the Hugging Face Space that proved
  the Gemma 4 E2B QAT mobile-transformers WebGPU path in a static browser demo:
  <https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels>
- `google/gemma-4-E2B-it-qat-mobile-transformers`, the model repository used by
  the demo and by this package at runtime:
  <https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers>
- Google DeepMind Gemma documentation and model cards.

The package does not vendor the Hugging Face Space bundle or model weights.

## License

Package code is MIT licensed.

Model weights, model cards, and upstream examples are governed by their own
licenses and terms. At the time this package was created, the Hugging Face model
card for `google/gemma-4-E2B-it-qat-mobile-transformers` listed Apache-2.0.
Review the model repository before shipping an application that downloads or
runs the model.
