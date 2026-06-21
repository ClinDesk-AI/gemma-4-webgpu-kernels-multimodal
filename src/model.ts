import { Gemma4AudioFeedForwardBlockName, Gemma4AudioFeedForwardProjectionName, Gemma4AudioLightConvProjectionName, Gemma4AudioSelfAttentionProjectionName, Gemma4VisionAttentionProjectionName, Gemma4VisionMlpProjectionName, QatProjectionTensorNames, VisionEncoderLayerNormTensorNames } from "./types.js";

export const GEMMA4_MEDIA_MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers";

export const HF_RESOLVE_BASE = "https://huggingface.co";

export const MEDIA_METADATA_CACHE_NAME = "clindesk-gemma4-media-metadata-v1";

export const MEDIA_TENSOR_CACHE_NAME = "clindesk-gemma4-media-tensor-slices-v1";

export const MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024;

export const GPU_BUFFER_USAGE_MAP_READ = 0x0001;

export const GPU_BUFFER_USAGE_COPY_SRC = 0x0004;

export const GPU_BUFFER_USAGE_COPY_DST = 0x0008;

export const GPU_BUFFER_USAGE_UNIFORM = 0x0040;

export const GPU_BUFFER_USAGE_STORAGE = 0x0080;

export const GPU_MAP_MODE_READ = 0x0001;

export const VISION_ATTENTION_PROJECTION_TENSORS = {
  q_proj: {
    weight: "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
    weightScale: "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_scale",
    inputScale: "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.input_activation_scale",
    outputScale: "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.output_activation_scale",
  },
  k_proj: {
    weight: "model.vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight",
    weightScale: "model.vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight_scale",
    inputScale: "model.vision_tower.encoder.layers.0.self_attn.k_proj.linear.input_activation_scale",
    outputScale: "model.vision_tower.encoder.layers.0.self_attn.k_proj.linear.output_activation_scale",
  },
  v_proj: {
    weight: "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight",
    weightScale: "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight_scale",
    inputScale: "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.input_activation_scale",
    outputScale: "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.output_activation_scale",
  },
} as const;

export const VISION_ATTENTION_NORM_TENSORS = {
  q_proj: "model.vision_tower.encoder.layers.0.self_attn.q_norm.weight",
  k_proj: "model.vision_tower.encoder.layers.0.self_attn.k_norm.weight",
  v_proj: null,
} as const;

export const VISION_ATTENTION_OUTPUT_PROJECTION_TENSORS = {
  weight: "model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight",
  weightScale: "model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight_scale",
  inputScale: "model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.input_activation_scale",
  outputScale: "model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.output_activation_scale",
} as const;

export const VISION_MLP_PROJECTION_TENSORS = {
  gate_proj: {
    weight: "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
    weightScale: "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight_scale",
    inputScale: "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.input_activation_scale",
    outputScale: "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.output_activation_scale",
  },
  up_proj: {
    weight: "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
    weightScale: "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight_scale",
    inputScale: "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.input_activation_scale",
    outputScale: "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.output_activation_scale",
  },
  down_proj: {
    weight: "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight",
    weightScale: "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight_scale",
    inputScale: "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.input_activation_scale",
    outputScale: "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.output_activation_scale",
  },
} as const;

export const VISION_ENCODER_LAYER0_NORM_TENSORS = {
  inputLayerNorm: "model.vision_tower.encoder.layers.0.input_layernorm.weight",
  postAttentionLayerNorm: "model.vision_tower.encoder.layers.0.post_attention_layernorm.weight",
  preFeedForwardLayerNorm: "model.vision_tower.encoder.layers.0.pre_feedforward_layernorm.weight",
  postFeedForwardLayerNorm: "model.vision_tower.encoder.layers.0.post_feedforward_layernorm.weight",
} as const;

export const VISION_ENCODER_LAYER_COUNT = 16;

export const AUDIO_ENCODER_LAYER_COUNT = 12;

export const AUDIO_FEED_FORWARD_RESIDUAL_WEIGHT = 0.5;

export function checkedVisionLayerIndex(layerIndex: number): number {
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= VISION_ENCODER_LAYER_COUNT) {
    throw new Error(`Vision encoder layer index must be 0-${VISION_ENCODER_LAYER_COUNT - 1}, got ${layerIndex}.`);
  }
  return layerIndex;
}

export function checkedVisionLayerCount(layerCount = VISION_ENCODER_LAYER_COUNT): number {
  if (!Number.isInteger(layerCount) || layerCount < 1 || layerCount > VISION_ENCODER_LAYER_COUNT) {
    throw new Error(`Vision encoder layer count must be 1-${VISION_ENCODER_LAYER_COUNT}, got ${layerCount}.`);
  }
  return layerCount;
}

export function visionAttentionProjectionTensors(
  layerIndex: number,
  projection: Gemma4VisionAttentionProjectionName,
): QatProjectionTensorNames {
  const index = checkedVisionLayerIndex(layerIndex);
  if (index === 0) return VISION_ATTENTION_PROJECTION_TENSORS[projection];
  const prefix = `model.vision_tower.encoder.layers.${index}.self_attn.${projection}.linear`;
  return {
    weight: `${prefix}.weight`,
    weightScale: `${prefix}.weight_scale`,
    inputScale: `${prefix}.input_activation_scale`,
    outputScale: `${prefix}.output_activation_scale`,
  };
}

export function visionAttentionNormTensor(
  layerIndex: number,
  projection: Gemma4VisionAttentionProjectionName,
): string | null {
  const index = checkedVisionLayerIndex(layerIndex);
  if (index === 0) return VISION_ATTENTION_NORM_TENSORS[projection];
  if (projection === "v_proj") return null;
  return `model.vision_tower.encoder.layers.${index}.self_attn.${projection === "q_proj" ? "q" : "k"}_norm.weight`;
}

export function visionAttentionOutputProjectionTensors(
  layerIndex: number,
): QatProjectionTensorNames {
  const index = checkedVisionLayerIndex(layerIndex);
  if (index === 0) return VISION_ATTENTION_OUTPUT_PROJECTION_TENSORS;
  const prefix = `model.vision_tower.encoder.layers.${index}.self_attn.o_proj.linear`;
  return {
    weight: `${prefix}.weight`,
    weightScale: `${prefix}.weight_scale`,
    inputScale: `${prefix}.input_activation_scale`,
    outputScale: `${prefix}.output_activation_scale`,
  };
}

export function visionMlpProjectionTensors(
  layerIndex: number,
  projection: Gemma4VisionMlpProjectionName,
): QatProjectionTensorNames {
  const index = checkedVisionLayerIndex(layerIndex);
  if (index === 0) return VISION_MLP_PROJECTION_TENSORS[projection];
  const prefix = `model.vision_tower.encoder.layers.${index}.mlp.${projection}.linear`;
  return {
    weight: `${prefix}.weight`,
    weightScale: `${prefix}.weight_scale`,
    inputScale: `${prefix}.input_activation_scale`,
    outputScale: `${prefix}.output_activation_scale`,
  };
}

export function visionEncoderLayerNormTensors(layerIndex: number): VisionEncoderLayerNormTensorNames {
  const index = checkedVisionLayerIndex(layerIndex);
  if (index === 0) return VISION_ENCODER_LAYER0_NORM_TENSORS;
  return {
    inputLayerNorm: `model.vision_tower.encoder.layers.${index}.input_layernorm.weight`,
    postAttentionLayerNorm: `model.vision_tower.encoder.layers.${index}.post_attention_layernorm.weight`,
    preFeedForwardLayerNorm: `model.vision_tower.encoder.layers.${index}.pre_feedforward_layernorm.weight`,
    postFeedForwardLayerNorm: `model.vision_tower.encoder.layers.${index}.post_feedforward_layernorm.weight`,
  };
}

export function audioSubsampleTensorNames(): {
  layer0Conv: string;
  layer0Norm: string;
  layer1Conv: string;
  layer1Norm: string;
  inputProjection: string;
} {
  return {
    layer0Conv: "model.audio_tower.subsample_conv_projection.layer0.conv.weight",
    layer0Norm: "model.audio_tower.subsample_conv_projection.layer0.norm.weight",
    layer1Conv: "model.audio_tower.subsample_conv_projection.layer1.conv.weight",
    layer1Norm: "model.audio_tower.subsample_conv_projection.layer1.norm.weight",
    inputProjection: "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
  };
}

export function checkedAudioLayerIndex(layerIndex: number): number {
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= AUDIO_ENCODER_LAYER_COUNT) {
    throw new Error(`Audio encoder layer index must be 0-${AUDIO_ENCODER_LAYER_COUNT - 1}, got ${layerIndex}.`);
  }
  return layerIndex;
}

export function checkedAudioLayerCount(layerCount = AUDIO_ENCODER_LAYER_COUNT): number {
  if (!Number.isInteger(layerCount) || layerCount < 1 || layerCount > AUDIO_ENCODER_LAYER_COUNT) {
    throw new Error(`Audio encoder layer count must be 1-${AUDIO_ENCODER_LAYER_COUNT}, got ${layerCount}.`);
  }
  return layerCount;
}

export function audioFeedForwardNormTensors(
  layerIndex: number,
  block: Gemma4AudioFeedForwardBlockName,
): { preLayerNorm: string; postLayerNorm: string } {
  const index = checkedAudioLayerIndex(layerIndex);
  return {
    preLayerNorm: `model.audio_tower.layers.${index}.${block}.pre_layer_norm.weight`,
    postLayerNorm: `model.audio_tower.layers.${index}.${block}.post_layer_norm.weight`,
  };
}

export function audioFeedForwardProjectionTensors(
  layerIndex: number,
  block: Gemma4AudioFeedForwardBlockName,
  projection: Gemma4AudioFeedForwardProjectionName,
): QatProjectionTensorNames {
  const index = checkedAudioLayerIndex(layerIndex);
  const prefix = `model.audio_tower.layers.${index}.${block}.${projection}.linear`;
  return {
    weight: `${prefix}.weight`,
    weightScale: `${prefix}.weight_scale`,
    inputScale: `${prefix}.input_activation_scale`,
    outputScale: `${prefix}.output_activation_scale`,
  };
}

export function audioLightConvTensors(
  layerIndex: number,
): { preLayerNorm: string; convNorm: string; depthwiseConv: string } {
  const index = checkedAudioLayerIndex(layerIndex);
  const prefix = `model.audio_tower.layers.${index}.lconv1d`;
  return {
    preLayerNorm: `${prefix}.pre_layer_norm.weight`,
    convNorm: `${prefix}.conv_norm.weight`,
    depthwiseConv: `${prefix}.depthwise_conv1d.weight`,
  };
}

export function audioLightConvProjectionTensors(
  layerIndex: number,
  projection: Gemma4AudioLightConvProjectionName,
): QatProjectionTensorNames {
  const index = checkedAudioLayerIndex(layerIndex);
  const prefix = `model.audio_tower.layers.${index}.lconv1d.${projection}.linear`;
  return {
    weight: `${prefix}.weight`,
    weightScale: `${prefix}.weight_scale`,
    inputScale: `${prefix}.input_activation_scale`,
    outputScale: `${prefix}.output_activation_scale`,
  };
}

export function audioSelfAttentionProjectionTensors(
  layerIndex: number,
  projection: Gemma4AudioSelfAttentionProjectionName,
): QatProjectionTensorNames {
  const index = checkedAudioLayerIndex(layerIndex);
  const prefix = `model.audio_tower.layers.${index}.self_attn.${projection}.linear`;
  return {
    weight: `${prefix}.weight`,
    weightScale: `${prefix}.weight_scale`,
    inputScale: `${prefix}.input_activation_scale`,
    outputScale: `${prefix}.output_activation_scale`,
  };
}

export function audioSelfAttentionTensors(
  layerIndex: number,
): { perDimScale: string; relativeKeyProjection: string } {
  const index = checkedAudioLayerIndex(layerIndex);
  const prefix = `model.audio_tower.layers.${index}.self_attn`;
  return {
    perDimScale: `${prefix}.per_dim_scale`,
    relativeKeyProjection: `${prefix}.relative_k_proj.weight`,
  };
}

export function audioEncoderLayerNormTensors(
  layerIndex: number,
): { normPreAttention: string; normPostAttention: string; normOut: string } {
  const index = checkedAudioLayerIndex(layerIndex);
  const prefix = `model.audio_tower.layers.${index}`;
  return {
    normPreAttention: `${prefix}.norm_pre_attn.weight`,
    normPostAttention: `${prefix}.norm_post_attn.weight`,
    normOut: `${prefix}.norm_out.weight`,
  };
}

export function audioOutputProjectionTensors(): { weight: string; bias: string } {
  return {
    weight: "model.audio_tower.output_proj.weight",
    bias: "model.audio_tower.output_proj.bias",
  };
}

export const GEMMA4_MEDIA_KERNEL_PENDING_MESSAGE =
  "Image and audio support must be implemented against google/gemma-4-E2B-it-qat-mobile-transformers.";