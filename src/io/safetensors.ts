import { MAX_SAFETENSORS_HEADER_BYTES, MEDIA_METADATA_CACHE_NAME, MEDIA_TENSOR_CACHE_NAME } from "../model.js";
import { Gemma4MediaTensorSliceProbe, JsonRecord, SafetensorsHeader, SafetensorsTensorByteRange, SafetensorsTensorData, SafetensorsTensorInfo } from "../types.js";
import { checksumBytes } from "../utils/math.js";

export async function loadSafetensorsTensorSlice(
  url: string,
  header: SafetensorsHeader,
  name: string,
  signal?: AbortSignal,
): Promise<Gemma4MediaTensorSliceProbe["tensors"][number]> {
  const data = await loadSafetensorsTensorData(url, header, name, signal);
  return {
    name: data.name,
    dtype: data.dtype,
    shape: data.shape,
    dataBytes: data.dataBytes,
    absoluteStart: data.absoluteStart,
    absoluteEndInclusive: data.absoluteEndInclusive,
    fromCache: data.fromCache,
    checksum: checksumBytes(data.bytes),
  };
}

export async function loadSafetensorsTensorData(
  url: string,
  header: SafetensorsHeader,
  name: string,
  signal?: AbortSignal,
): Promise<SafetensorsTensorData> {
  const info = header.tensors[name];
  if (!info) throw new Error(`Safetensors tensor not found: ${name}`);
  const [relativeStart, relativeEnd] = info.data_offsets;
  const dataBytes = relativeEnd - relativeStart;
  if (dataBytes <= 0) throw new Error(`Safetensors tensor has invalid data range: ${name}`);

  const absoluteStart = 8 + header.headerBytes + relativeStart;
  const absoluteEndInclusive = 8 + header.headerBytes + relativeEnd - 1;
  const cacheKey = tensorCacheRequest(url, name, absoluteStart, absoluteEndInclusive);
  const cache = await openTensorCache();
  const cached = await cache?.match(cacheKey).catch(() => undefined);
  if (cached) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    return {
      name,
      dtype: info.dtype,
      shape: info.shape,
      dataBytes,
      absoluteStart,
      absoluteEndInclusive,
      fromCache: true,
      bytes,
    };
  }

  const buffer = await fetchRange(url, absoluteStart, absoluteEndInclusive, signal);
  const bytes = new Uint8Array(buffer);
  if (bytes.byteLength !== dataBytes) {
    throw new Error(`Safetensors tensor range returned ${bytes.byteLength} bytes, expected ${dataBytes}: ${name}`);
  }
  await cache?.put(cacheKey, new Response(buffer.slice(0), {
    headers: {
      "content-type": "application/octet-stream",
      "x-clindesk-source": url,
      "x-clindesk-tensor": name,
    },
  })).catch(() => undefined);
  return {
    name,
    dtype: info.dtype,
    shape: info.shape,
    dataBytes,
    absoluteStart,
    absoluteEndInclusive,
    fromCache: false,
    bytes,
  };
}

export async function loadSafetensorsTensorByteRange(
  url: string,
  header: SafetensorsHeader,
  name: string,
  relativeByteOffset: number,
  byteLength: number,
  signal?: AbortSignal,
): Promise<SafetensorsTensorByteRange> {
  const info = header.tensors[name];
  if (!info) throw new Error(`Safetensors tensor not found: ${name}`);
  if (!Number.isInteger(relativeByteOffset) || relativeByteOffset < 0) {
    throw new Error(`Invalid tensor byte offset ${relativeByteOffset}: ${name}`);
  }
  if (!Number.isInteger(byteLength) || byteLength <= 0) {
    throw new Error(`Invalid tensor byte length ${byteLength}: ${name}`);
  }
  const [relativeStart, relativeEnd] = info.data_offsets;
  const tensorBytes = relativeEnd - relativeStart;
  if (relativeByteOffset + byteLength > tensorBytes) {
    throw new Error(`Tensor byte range exceeds tensor size: ${name}`);
  }

  const absoluteStart = 8 + header.headerBytes + relativeStart + relativeByteOffset;
  const absoluteEndInclusive = absoluteStart + byteLength - 1;
  const cacheKey = tensorCacheRequest(url, `${name}@${relativeByteOffset}:${byteLength}`, absoluteStart, absoluteEndInclusive);
  const cache = await openTensorCache();
  const cached = await cache?.match(cacheKey).catch(() => undefined);
  if (cached) {
    return {
      name,
      dataBytes: byteLength,
      absoluteStart,
      absoluteEndInclusive,
      fromCache: true,
      bytes: new Uint8Array(await cached.arrayBuffer()),
    };
  }

  const buffer = await fetchRange(url, absoluteStart, absoluteEndInclusive, signal);
  const bytes = new Uint8Array(buffer);
  if (bytes.byteLength !== byteLength) {
    throw new Error(`Safetensors tensor range returned ${bytes.byteLength} bytes, expected ${byteLength}: ${name}`);
  }
  await cache?.put(cacheKey, new Response(buffer.slice(0), {
    headers: {
      "content-type": "application/octet-stream",
      "x-clindesk-source": url,
      "x-clindesk-tensor": name,
    },
  })).catch(() => undefined);
  return {
    name,
    dataBytes: byteLength,
    absoluteStart,
    absoluteEndInclusive,
    fromCache: false,
    bytes,
  };
}

export function float32FromBytes(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`F32 tensor byte length must be a multiple of 4, got ${bytes.byteLength}.`);
  }
  const values = new Float32Array(bytes.byteLength / 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getFloat32(index * 4, true);
  }
  return values;
}

export function float32FromBf16Bytes(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 2 !== 0) {
    throw new Error(`BF16 tensor byte length must be a multiple of 2, got ${bytes.byteLength}.`);
  }
  const values = new Float32Array(bytes.byteLength / 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const scratch = new Uint32Array(1);
  const scratchFloat = new Float32Array(scratch.buffer);
  for (let index = 0; index < values.length; index += 1) {
    scratch[0] = view.getUint16(index * 2, true) << 16;
    values[index] = scratchFloat[0];
  }
  return values;
}

export function int8FromBytes(bytes: Uint8Array): Int8Array {
  return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

export function scalarF32FromTensor(tensor: SafetensorsTensorData, name: string): number {
  if (tensor.dtype !== "F32" || tensor.shape.length !== 0 || tensor.bytes.byteLength !== 4) {
    throw new Error(`Expected scalar F32 tensor for ${name}, got ${tensor.dtype} [${tensor.shape.join(", ")}].`);
  }
  return float32FromBytes(tensor.bytes)[0];
}

export function assertBf16Vector(tensor: SafetensorsTensorData, length: number, label: string): void {
  if (tensor.dtype !== "BF16" || tensor.shape.length !== 1 || tensor.shape[0] !== length) {
    throw new Error(`Unexpected ${label} tensor: ${tensor.dtype} [${tensor.shape.join(", ")}].`);
  }
}

export function assertF32Tensor(tensor: SafetensorsTensorData, shape: number[], label: string): void {
  if (tensor.dtype !== "F32" ||
    tensor.shape.length !== shape.length ||
    tensor.shape.some((value, index) => value !== shape[index])
  ) {
    throw new Error(`Unexpected ${label} tensor: ${tensor.dtype} [${tensor.shape.join(", ")}].`);
  }
}

export function tensorCacheRequest(
  sourceUrl: string,
  name: string,
  absoluteStart: number,
  absoluteEndInclusive: number,
): Request {
  const origin = globalThis.location?.origin ?? "https://app.clindesk.ai";
  const url = new URL("/__clindesk_cache__/gemma4-media-tensor", origin);
  url.searchParams.set("source", sourceUrl);
  url.searchParams.set("name", name);
  url.searchParams.set("start", String(absoluteStart));
  url.searchParams.set("end", String(absoluteEndInclusive));
  return new Request(url.toString());
}

export async function fetchJson(url: string, signal?: AbortSignal): Promise<JsonRecord> {
  const source = await loadCachedText(url, signal);
  const parsed = JSON.parse(source) as unknown;
  return record(parsed);
}

export async function loadCachedText(url: string, signal?: AbortSignal): Promise<string> {
  const cache = await openMetadataCache();
  const cached = await cache?.match(url).catch(() => undefined);
  if (cached) return cached.text();

  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Request failed with ${response.status}: ${url}`);
  const text = await response.text();
  await cache?.put(url, new Response(text, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  })).catch(() => undefined);
  return text;
}

export async function openMetadataCache(): Promise<Cache | null> {
  if (!("caches" in globalThis)) return null;
  try {
    return await caches.open(MEDIA_METADATA_CACHE_NAME);
  } catch {
    return null;
  }
}

export async function openTensorCache(): Promise<Cache | null> {
  if (!("caches" in globalThis)) return null;
  try {
    return await caches.open(MEDIA_TENSOR_CACHE_NAME);
  } catch {
    return null;
  }
}

export async function fetchSafetensorsHeader(
  url: string,
  signal?: AbortSignal,
): Promise<SafetensorsHeader> {
  const cache = await openMetadataCache();
  const cacheKey = safetensorsHeaderCacheRequest(url);
  const cached = await cache?.match(cacheKey).catch(() => undefined);
  if (cached) {
    try {
      const parsed = record(JSON.parse(await cached.text()));
      const headerBytes = Number(parsed.headerBytes);
      const tensorRecords = record(parsed.tensors);
      if (Number.isInteger(headerBytes) && headerBytes > 0) {
        const tensors = Object.fromEntries(
          Object.entries(tensorRecords).map(([name, value]) => [
            name,
            safetensorsTensorInfo(name, value),
          ]),
        );
        return { headerBytes, tensors };
      }
    } catch {
      // Ignore malformed cache entries and refresh from the source range.
    }
  }

  const prefix = await fetchRange(url, 0, 7, signal);
  if (prefix.byteLength !== 8) {
    throw new Error(`Expected 8-byte safetensors header prefix, got ${prefix.byteLength} bytes.`);
  }

  const headerBytesBigInt = readUint64LittleEndian(new Uint8Array(prefix));
  if (headerBytesBigInt > BigInt(MAX_SAFETENSORS_HEADER_BYTES)) {
    throw new Error(`Safetensors header is too large: ${headerBytesBigInt.toString()} bytes.`);
  }

  const headerBytes = Number(headerBytesBigInt);
  const headerBuffer = await fetchRange(url, 8, 7 + headerBytes, signal);
  const header = JSON.parse(new TextDecoder().decode(headerBuffer)) as unknown;
  const headerRecord = record(header);
  const tensors = Object.fromEntries(
    Object.entries(headerRecord)
      .filter(([name]) => name !== "__metadata__")
      .map(([name, value]) => [name, safetensorsTensorInfo(name, value)]),
  );
  const result = { headerBytes, tensors };
  await cache?.put(cacheKey, new Response(JSON.stringify(result), {
    headers: {
      "content-type": "application/json",
      "x-clindesk-source": url,
      "x-clindesk-kind": "safetensors-header",
    },
  })).catch(() => undefined);
  return result;
}

function safetensorsHeaderCacheRequest(sourceUrl: string): Request {
  const origin = globalThis.location?.origin ?? "https://app.clindesk.ai";
  const url = new URL("/__clindesk_cache__/gemma4-media-header", origin);
  url.searchParams.set("source", sourceUrl);
  return new Request(url.toString());
}

export function safetensorsTensorInfo(name: string, value: unknown): SafetensorsTensorInfo {
  const data = record(value);
  const dtype = typeof data.dtype === "string" ? data.dtype : "";
  const shape = Array.isArray(data.shape)
    ? data.shape.map((part) => Number(part)).filter((part) => Number.isInteger(part) && part >= 0)
    : [];
  const offsets = Array.isArray(data.data_offsets) ? data.data_offsets.map(Number) : [];
  if (!dtype || offsets.length !== 2 || !offsets.every((part) => Number.isInteger(part) && part >= 0)) {
    throw new Error(`Invalid safetensors metadata for ${name}`);
  }
  return {
    dtype,
    shape,
    data_offsets: [offsets[0], offsets[1]],
  };
}

export async function fetchRange(
  url: string,
  start: number,
  end: number,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const response = await fetch(url, {
    headers: { Range: `bytes=${start}-${end}` },
    signal,
  });
  if (response.status !== 206) {
    throw new Error(`Range request failed with ${response.status}; refusing to download full model.safetensors.`);
  }
  return response.arrayBuffer();
}

export function readUint64LittleEndian(bytes: Uint8Array): bigint {
  let value = 0n;
  for (let index = 0; index < bytes.length; index += 1) {
    value += BigInt(bytes[index]) << (8n * BigInt(index));
  }
  return value;
}

export function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : {};
}

export function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
