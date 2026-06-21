export {
  GEMMA4_MEDIA_KERNEL_PENDING_MESSAGE,
} from "./model.js";

export {
  computeGemma4AudioEmbeddings,
} from "./audio/index.js";

export {
  computeGemma4AudioSoftTokenCount,
  preprocessGemma4AudioSamples,
} from "./preprocess/audio.js";

export {
  computeGemma4ImageTokenLayout,
  preprocessGemma4ImageBlob,
} from "./preprocess/image.js";

export {
  computeGemma4VisionImageEmbeddings,
} from "./vision/index.js";

export type {
  Gemma4AudioEmbeddingResult,
  Gemma4AudioPreprocessOptions,
  Gemma4AudioPreprocessProbe,
  Gemma4AudioPreprocessResult,
  Gemma4ImagePreprocessOptions,
  Gemma4ImagePreprocessProbe,
  Gemma4ImagePreprocessResult,
  Gemma4ImageTokenLayout,
  Gemma4VisionImageEmbeddingResult,
} from "./types.js";

export const GEMMA4_MEDIA_MODEL_ID =
  "google/gemma-4-E2B-it-qat-mobile-transformers";

export const GEMMA4_MEDIA_MODEL_URL =
  "https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers";

export const UPSTREAM_WEBGPU_KERNEL_SPACE_URL =
  "https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels";
