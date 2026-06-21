import { readdirSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const indexSource = readFileSync(resolve(root, "src/index.ts"), "utf8");

function sourceFiles(dir) {
  return readdirSync(dir)
    .flatMap((entry) => {
      const path = resolve(dir, entry);
      const stat = statSync(path);
      if (stat.isDirectory()) return sourceFiles(path);
      return path.endsWith(".ts") ? [path] : [];
    });
}

const files = sourceFiles(resolve(root, "src"));
const sourceByPath = new Map(
  files.map((path) => [path, readFileSync(path, "utf8")]),
);
const fullSource = [...sourceByPath.values()].join("\n");

const requiredIndexExports = [
  "computeGemma4VisionImageEmbeddings",
  "computeGemma4AudioEmbeddings",
  "preprocessGemma4ImageBlob",
  "preprocessGemma4AudioSamples",
  "GEMMA4_MEDIA_MODEL_ID",
];

const requiredCoreSignals = [
  "google/gemma-4-E2B-it-qat-mobile-transformers",
  "model.vision_tower",
  "model.audio_tower",
  "navigatorGpu",
  "caches.open",
];

const requiredFiles = [
  "src/audio/encoder.ts",
  "src/audio/kernels.ts",
  "src/audio/probes.ts",
  "src/embedder/index.ts",
  "src/gpu/runtime.ts",
  "src/io/safetensors.ts",
  "src/kernels/qat.ts",
  "src/preprocess/audio.ts",
  "src/preprocess/image.ts",
  "src/vision/encoder.ts",
  "src/vision/kernels.ts",
  "src/vision/probes.ts",
];

const missingIndexExports = requiredIndexExports.filter(
  (needle) => !indexSource.includes(needle),
);
const missingCoreSignals = requiredCoreSignals.filter(
  (needle) => !fullSource.includes(needle),
);
const missingFiles = requiredFiles.filter(
  (relativePath) => !sourceByPath.has(resolve(root, relativePath)),
);
const oversizedFiles = [...sourceByPath.entries()].filter(
  ([, source]) => source.split("\n").length > 2_000,
);

if (
  missingIndexExports.length > 0 ||
  missingCoreSignals.length > 0 ||
  missingFiles.length > 0 ||
  oversizedFiles.length > 0
) {
  console.error("Static probe failed.");
  if (missingIndexExports.length > 0) {
    console.error(`Missing index exports: ${missingIndexExports.join(", ")}`);
  }
  if (missingCoreSignals.length > 0) {
    console.error(`Missing core signals: ${missingCoreSignals.join(", ")}`);
  }
  if (missingFiles.length > 0) {
    console.error(`Missing split files: ${missingFiles.join(", ")}`);
  }
  if (oversizedFiles.length > 0) {
    console.error(
      `Oversized files: ${oversizedFiles
        .map(([path, source]) => `${path.replace(`${root}/`, "")} (${source.split("\n").length} lines)`)
        .join(", ")}`,
    );
  }
  process.exit(1);
}

console.log("Static probe passed.");
