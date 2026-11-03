/**
 * Per-repo configuration for pi-annotations, read from
 * <repoRoot>/.pi/annotations/config.json (next to the notes, so it travels
 * with the repo). pi's own settings.json schema has no extension bag, so the
 * repo-local file is the config home. Everything is optional; absent file or
 * absent keys mean defaults, and unknown keys are ignored.
 */

import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { ANNOTATIONS_DIR } from "./notes.ts";

export const CONFIG_FILE = "config.json";

/** Absolute path of <repoRoot>/.pi/annotations/config.json. */
export function configFilePath(repoRoot: string): string {
	return join(repoRoot, ANNOTATIONS_DIR, CONFIG_FILE);
}

/** Gate backend preference. "auto" tries local backends first, then chat. */
export type GateBackend = "auto" | "python" | "llama.cpp" | "api" | "chat" | "off";

export interface RelevanceGateConfig {
	/** Master switch. Default false: injection behaves exactly as without the gate. */
	enabled: boolean;
	/** Which backend runs the gate. "auto" = python → llama.cpp → api → chat. */
	backend: GateBackend;
	/** Chat-backend model override as "provider/modelId" (default: current session model). */
	model?: string;
	/** Local engine the python backend was set up with / should use: "julia-1" | "laya". */
	localModel?: string;
	/** Classifier model override for llama.cpp/api backends, as "provider/modelId" or bare id. */
	classifierModel?: string;
	/** Max candidates sent in one gate call; the rest are kept unjudged (fail-open). */
	maxCandidates: number;
	/** Below this many candidate notes the gate is skipped (a lone note is bounded by its size). */
	minItemsToGate: number;
	/** Hard wall-clock cap for one gate call; on expiry all candidates are kept. */
	timeoutMs: number;
	/** Classifier backends: keep when P(relevant) >= keepThreshold. */
	keepThreshold: number;
	/** Classifier backends: |P - 0.5| below this band counts as "unsure" → keep (fail-open). */
	ambiguityBand: number;
	/** Cap on extracted task text sent to the gate (classifier state budgets are ~1k tokens). */
	taskMaxChars: number;
}

export interface AnnotationsConfig {
	relevanceGate: RelevanceGateConfig;
}

export const DEFAULT_CONFIG: AnnotationsConfig = {
	relevanceGate: {
		enabled: false,
		backend: "auto",
		maxCandidates: 20,
		minItemsToGate: 2,
		timeoutMs: 5000,
		keepThreshold: 0.5,
		ambiguityBand: 0.1,
		taskMaxChars: 2000,
	},
};

function num(v: unknown, fallback: number, min: number, max: number): number {
	if (typeof v !== "number" || !Number.isFinite(v)) return fallback;
	return Math.min(max, Math.max(min, v));
}

function bool(v: unknown, fallback: boolean): boolean {
	return typeof v === "boolean" ? v : fallback;
}

function backend(v: unknown): GateBackend {
	return v === "auto" || v === "python" || v === "llama.cpp" || v === "api" || v === "chat" || v === "off" ? v : "auto";
}

/** Parse raw JSON into a config, filling defaults and clamping ranges. Never throws. */
export function parseConfig(raw: unknown): AnnotationsConfig {
	const gate = (raw && typeof raw === "object" ? (raw as any).relevanceGate : undefined) ?? {};
	const d = DEFAULT_CONFIG.relevanceGate;
	const model = typeof gate.model === "string" && gate.model.includes("/") ? gate.model : undefined;
	const localModel = gate.localModel === "julia-1" || gate.localModel === "laya" ? gate.localModel : undefined;
	const classifierModel = typeof gate.classifierModel === "string" && gate.classifierModel ? gate.classifierModel : undefined;
	return {
		relevanceGate: {
			enabled: bool(gate.enabled, d.enabled),
			backend: backend(gate.backend),
			model,
			localModel,
			classifierModel,
			maxCandidates: Math.round(num(gate.maxCandidates, d.maxCandidates, 1, 100)),
			minItemsToGate: Math.round(num(gate.minItemsToGate, d.minItemsToGate, 1, 100)),
			timeoutMs: Math.round(num(gate.timeoutMs, d.timeoutMs, 100, 60_000)),
			keepThreshold: num(gate.keepThreshold, d.keepThreshold, 0, 1),
			ambiguityBand: num(gate.ambiguityBand, d.ambiguityBand, 0, 0.5),
			taskMaxChars: Math.round(num(gate.taskMaxChars, d.taskMaxChars, 200, 20_000)),
		},
	};
}

interface CacheEntry {
	mtimeMs: number;
	config: AnnotationsConfig;
}
const cache = new Map<string, CacheEntry>();

/** Load and cache <repoRoot>/.pi/annotations/config.json (re-read when the file changes). */
export async function loadConfig(repoRoot: string): Promise<AnnotationsConfig> {
	const file = configFilePath(repoRoot);
	try {
		const st = await stat(file);
		const hit = cache.get(repoRoot);
		if (hit && hit.mtimeMs === st.mtimeMs) return hit.config;
		const raw = JSON.parse(await readFile(file, "utf-8"));
		const config = parseConfig(raw);
		cache.set(repoRoot, { mtimeMs: st.mtimeMs, config });
		return config;
	} catch {
		// missing or malformed config → defaults; malformed JSON falls through to defaults too
		const hit = cache.get(repoRoot);
		if (hit) return hit.config;
		cache.set(repoRoot, { mtimeMs: -1, config: DEFAULT_CONFIG });
		return DEFAULT_CONFIG;
	}
}

/** Drop the cached config for a repo (e.g. right after writeConfig). */
export function invalidateConfigCache(repoRoot: string): void {
	cache.delete(repoRoot);
}

/**
 * Read the raw parsed JSON of the config file, so editors can rewrite only the keys
 * they manage and leave unknown keys untouched. Returns {} when missing/malformed.
 */
export async function readRawConfig(repoRoot: string): Promise<Record<string, unknown>> {
	try {
		const raw = JSON.parse(await readFile(configFilePath(repoRoot), "utf-8"));
		return raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};
	} catch {
		return {};
	}
}

/** Write the config file as pretty JSON, creating .pi/annotations when needed. */
export async function writeConfig(repoRoot: string, raw: unknown): Promise<void> {
	const file = configFilePath(repoRoot);
	await mkdir(dirname(file), { recursive: true });
	await writeFile(file, JSON.stringify(raw, null, 2) + "\n", "utf-8");
	invalidateConfigCache(repoRoot);
}

/** Serialize a parsed gate config back to plain JSON, omitting unset optional keys. */
export function gateConfigToRaw(gate: RelevanceGateConfig): Record<string, unknown> {
	const out: Record<string, unknown> = {
		enabled: gate.enabled,
		backend: gate.backend,
		maxCandidates: gate.maxCandidates,
		minItemsToGate: gate.minItemsToGate,
		timeoutMs: gate.timeoutMs,
		keepThreshold: gate.keepThreshold,
		ambiguityBand: gate.ambiguityBand,
		taskMaxChars: gate.taskMaxChars,
	};
	if (gate.model) out.model = gate.model;
	if (gate.localModel) out.localModel = gate.localModel;
	if (gate.classifierModel) out.classifierModel = gate.classifierModel;
	return out;
}
