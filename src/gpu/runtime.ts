import { NavigatorWithGpu } from "../types.js";

export function alignedBufferSize(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

export async function shaderCompilationMessages(module: GPUShaderModule): Promise<string[]> {
  const candidate = module as GPUShaderModule & {
    getCompilationInfo?: () => Promise<{
      messages: Array<{
        type?: string;
        message?: string;
        lineNum?: number;
        linePos?: number;
      }>;
    }>;
  };
  const info = await candidate.getCompilationInfo?.().catch(() => null);
  return (info?.messages ?? [])
    .filter((message) => message.type === "error")
    .map((message) => {
      const location = message.lineNum && message.linePos ? `${message.lineNum}:${message.linePos}` : "unknown";
      return `${location} ${message.message ?? "unknown shader error"}`;
    });
}

export function writeGpuBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  bufferOffset: number,
  data: ArrayBuffer | ArrayBufferView<ArrayBufferLike>,
  dataOffset?: number,
  size?: number,
): void {
  device.queue.writeBuffer(
    buffer,
    bufferOffset,
    data as GPUAllowSharedBufferSource,
    dataOffset,
    size,
  );
}

export function navigatorGpu(): NavigatorWithGpu["gpu"] | null {
  return (navigator as NavigatorWithGpu).gpu ?? null;
}
