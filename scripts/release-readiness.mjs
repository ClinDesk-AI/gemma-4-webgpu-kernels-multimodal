#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = parseArgs(process.argv.slice(2));
const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));

const build = run("pnpm", ["run", "build"]);
const staticProbe = run("pnpm", ["run", "probe:static"]);
const pack = run("pnpm", ["pack", "--dry-run", "--json"]);
const packReport = parsePackJson(pack.stdoutText);
const packedFiles = new Set((packReport.files ?? []).map((file) => file.path));
const git = inspectGitState();

const requiredPackedFiles = [
  "LICENSE",
  "NOTICE",
  "README.md",
  "package.json",
  "dist/index.js",
  "dist/index.d.ts",
  "dist/probes/index.js",
  "dist/probes/index.d.ts",
  "dist/vision/index.js",
  "dist/audio/index.js",
  "dist/embedder/index.js",
];

const forbiddenPackedPatterns = [
  /^src\//,
  /^examples\//,
  /^scripts\//,
  /^node_modules\//,
  /^\.git\//,
  /^\.release\//,
  /\.(?:safetensors|onnx|gguf|bin|wasm)$/i,
];

const missingPackedFiles = requiredPackedFiles.filter((file) => !packedFiles.has(file));
const forbiddenPackedFiles = [...packedFiles].filter((file) =>
  forbiddenPackedPatterns.some((pattern) => pattern.test(file)),
);
const packageManager = String(packageJson.packageManager ?? "");
const checks = [
  check("package-name", packageJson.name === "@clindesk/gemma-4-webgpu-kernels-multimodal", packageJson.name),
  check("public-package", packageJson.private !== true, packageJson.private ?? false),
  check("pnpm-package-manager", packageManager.startsWith("pnpm@"), packageManager),
  check("public-publish-config", packageJson.publishConfig?.access === "public", packageJson.publishConfig),
  check("root-export-has-types", packageJson.exports?.["."]?.types === "./dist/index.d.ts", packageJson.exports?.["."]),
  check("root-export-has-default", packageJson.exports?.["."]?.default === "./dist/index.js", packageJson.exports?.["."]),
  check("probes-export-has-types", packageJson.exports?.["./probes"]?.types === "./dist/probes/index.d.ts", packageJson.exports?.["./probes"]),
  check("probes-export-has-default", packageJson.exports?.["./probes"]?.default === "./dist/probes/index.js", packageJson.exports?.["./probes"]),
  check("build-passes", build.ok, build.summary),
  check("static-probe-passes", staticProbe.ok, staticProbe.summary),
  check("pack-dry-run-passes", pack.ok && Boolean(packReport.name), pack.summary),
  check("pack-name-matches-package", packReport.name === packageJson.name, packReport.name),
  check("pack-version-matches-package", packReport.version === packageJson.version, packReport.version),
  check("pack-includes-required-files", missingPackedFiles.length === 0, missingPackedFiles),
  check("pack-excludes-source-and-artifacts", forbiddenPackedFiles.length === 0, forbiddenPackedFiles),
  check("git-working-tree-clean", git.clean === true, git.status),
  check("git-branch-main", git.branch === "main", git.branch),
  check("git-origin-is-clindesk-repo", git.originUrl === "https://github.com/ClinDesk-AI/gemma-4-webgpu-kernels-multimodal.git", git.originUrl),
  check("git-head-pushed-to-upstream", git.upstream === "origin/main" && git.ahead === 0, {
    upstream: git.upstream,
    ahead: git.ahead,
    behind: git.behind,
  }),
];

const report = {
  ok: checks.every((entry) => entry.ok),
  generatedAt: new Date().toISOString(),
  productSurface: false,
  package: {
    name: packageJson.name,
    version: packageJson.version,
    packageManager,
    publishConfig: packageJson.publishConfig ?? null,
  },
  commands: {
    build: commandEvidence(build),
    staticProbe: commandEvidence(staticProbe),
    pack: commandEvidence(pack),
  },
  pack: {
    name: packReport.name ?? null,
    version: packReport.version ?? null,
    filename: packReport.filename ?? null,
    fileCount: packedFiles.size,
    requiredPackedFiles,
    missingPackedFiles,
    forbiddenPackedFiles,
  },
  git,
  checks,
};

if (args.out) {
  const outPath = resolve(root, args.out);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
}

console.log(JSON.stringify(report, null, 2));
process.exitCode = report.ok ? 0 : 1;

function check(id, ok, evidence) {
  return { id, ok: Boolean(ok), evidence };
}

function run(command, commandArgs) {
  const result = spawnSync(command, commandArgs, {
    cwd: root,
    encoding: "utf8",
    timeout: 120_000,
  });
  return {
    ok: result.status === 0,
    command: [command, ...commandArgs].join(" "),
    exitCode: result.status,
    signal: result.signal ?? null,
    stdoutText: result.stdout,
    stdoutTail: tail(result.stdout),
    stderrTail: tail(result.stderr),
    summary: tail(result.stderr || result.stdout),
    stderr: tail(result.stderr),
  };
}

function commandEvidence(result) {
  return {
    ok: result.ok,
    command: result.command,
    exitCode: result.exitCode,
    signal: result.signal,
    stdoutTail: result.stdoutTail,
    stderrTail: result.stderrTail,
    summary: result.summary,
  };
}

function inspectGitState() {
  const status = run("git", ["status", "--porcelain=v1"]);
  const branch = run("git", ["rev-parse", "--abbrev-ref", "HEAD"]);
  const head = run("git", ["rev-parse", "HEAD"]);
  const origin = run("git", ["remote", "get-url", "origin"]);
  const upstream = run("git", ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]);
  const ahead = upstream.ok ? run("git", ["rev-list", "--count", "@{u}..HEAD"]) : null;
  const behind = upstream.ok ? run("git", ["rev-list", "--count", "HEAD..@{u}"]) : null;
  return {
    clean: status.ok && status.stdoutText.trim().length === 0,
    status: status.stdoutText.trim(),
    branch: branch.ok ? branch.stdoutText.trim() : "",
    head: head.ok ? head.stdoutText.trim() : "",
    originUrl: origin.ok ? origin.stdoutText.trim() : "",
    upstream: upstream.ok ? upstream.stdoutText.trim() : "",
    ahead: ahead?.ok ? Number(ahead.stdoutText.trim()) : null,
    behind: behind?.ok ? Number(behind.stdoutText.trim()) : null,
  };
}

function parsePackJson(stdout) {
  const text = String(stdout ?? "");
  const start = text.indexOf("{");
  if (start < 0) return {};
  try {
    return JSON.parse(text.slice(start));
  } catch {
    return {};
  }
}

function tail(text) {
  return String(text ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-8);
}

function parseArgs(argv) {
  const parsed = { out: ".release/latest-release-readiness.json" };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--") continue;
    if (arg === "--out") parsed.out = requireValue(argv, ++index, arg);
    else if (arg === "--help" || arg === "-h") {
      console.log([
        "Usage:",
        "  pnpm probe:release",
        "  pnpm probe:release -- --out .release/latest-release-readiness.json",
        "",
        "Builds, runs static checks, and verifies pnpm pack contents before OTP-gated publishing.",
      ].join("\n"));
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function requireValue(argv, index, flag) {
  const value = argv[index];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`${flag} requires a value.`);
  }
  return value;
}
