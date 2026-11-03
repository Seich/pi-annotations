/**
 * Task-relevance gate for injected annotations.
 *
 * The injectors (read/grep/shell) mechanically attach every note whose anchor
 * touches the shown lines. This module filters those candidates by relevance
 * to the current task before they reach the model:
 *
 *  - Task text is extracted from the session branch (last compaction summary,
 *    first + latest user message, tail of the latest assistant message) and
 *    cached per leaf id.
 *  - One batched gate call judges all candidates of a tool result at once.
 *    Backends are pluggable (GateBackendImpl); only "chat" is implemented here —
 *    local classifier backends (python/llama.cpp/api) plug in behind "auto".
 *  - Verdicts are cached per (task, note id, text, location, stale) so repeated
 *    reads of the same file within one task cost no extra calls.
 *  - Fail-open everywhere: any error, timeout, parse failure, missing model, or
 *    classifier ambiguity keeps the note. The gate can only ever REMOVE notes
 *    the model would otherwise see, and only when it is confident.
 *
 * Debug: PI_ANNOTATIONS_DEBUG=1 logs gate decisions to stderr.
 */

import { createHash } from "node:crypto";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { loadConfig, type RelevanceGateConfig } from "./config.ts";
import { AUTO_ORDER, GATE_BACKENDS } from "./backends.ts";
import type { AnnotationItem } from "./notes.ts";

// chat-backend prompt helpers live in backends.ts; re-exported for tests/consumers
export { buildGatePrompt, parseKeepJson } from "./backends.ts";

const DEBUG = !!process.env.PI_ANNOTATIONS_DEBUG;
function debug(...args: unknown[]): void {
	if (DEBUG) console.error("[pi-annotations:gate]", ...args);
}

function sha256(s: string): string {
	return createHash("sha256").update(s).digest("hex");
}

/** A candidate note as seen by the gate (AnnotationItem plus the note id for cache keys). */
export type GateCandidate = AnnotationItem & { id: string };

export interface GateRequest {
	task: string;
	tool: string;
	path?: string;
	candidates: GateCandidate[];
}

export interface GateOutcome {
	/** Kept items (same objects as input, filtered). */
	items: GateCandidate[];
	hidden: number;
	hiddenStale: number;
	/** "skipped" | "fail-open" | backend name that decided. */
	backend: string;
	latencyMs: number;
}

// ---------------------------------------------------------------- task extraction

function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		return content
			.filter((c: any) => c && typeof c === "object" && c.type === "text" && typeof c.text === "string")
			.map((c: any) => c.text as string)
			.join("\n");
	}
	return "";
}

/**
 * Build the "task at hand" string from raw session entries. Pure and exported
 * for testing. Uses the last compaction summary (covers pre-compaction
 * history), the first user message (the original ask), the latest user
 * message (task pivots), and the tail of the latest assistant message (the
 * current sub-goal). Truncated to maxChars — classifier state budgets are
 * ~1k tokens, so this must stay small.
 */
export function extractTaskText(entries: unknown[], maxChars: number): string {
	let firstUser = "";
	let lastUser = "";
	let lastSummary = "";
	let lastAssistantTail = "";
	for (const e of entries ?? []) {
		if (!e || typeof e !== "object") continue;
		const entry = e as any;
		if (entry.type === "compaction" && typeof entry.summary === "string" && entry.summary.trim()) {
			lastSummary = entry.summary;
			continue;
		}
		if (entry.type !== "message") continue;
		const m = entry.message;
		if (!m || typeof m !== "object") continue;
		if (m.role === "user") {
			const t = contentText(m.content).trim();
			if (t) {
				if (!firstUser) firstUser = t;
				lastUser = t;
			}
		} else if (m.role === "assistant") {
			const t = contentText(m.content).trim();
			if (t) lastAssistantTail = t.slice(-500);
		}
	}
	const parts: string[] = [];
	for (const p of [lastSummary, firstUser, lastUser, lastAssistantTail]) {
		if (p && !parts.includes(p)) parts.push(p);
	}
	return parts.join("\n\n").slice(0, maxChars);
}

let taskCache: { leafId: string | null; text: string } | null = null;

async function getTaskText(ctx: ExtensionContext, maxChars: number): Promise<string> {
	const sm = ctx.sessionManager;
	const leafId = sm.getLeafId();
	if (taskCache && taskCache.leafId === leafId) return taskCache.text;
	const text = extractTaskText(sm.getBranch(), maxChars);
	taskCache = { leafId, text };
	return text;
}

// ---------------------------------------------------------------- backends

/** judge() may return a bare boolean[] or { keep, raw } — raw is the backend's unparsed reply, kept for status/debug display. */
export interface GateJudgeResult {
	keep: boolean[];
	raw?: unknown;
}

/** One gate backend. judge() returns one keep/drop boolean per candidate, or throws (caller fails open). */
export interface GateBackendImpl {
	name: string;
	available(ctx: ExtensionContext, cfg: RelevanceGateConfig): boolean | Promise<boolean>;
	judge(ctx: ExtensionContext, cfg: RelevanceGateConfig, req: GateRequest, signal: AbortSignal): Promise<boolean[] | GateJudgeResult>;
}

/** Ordered backends to try: "auto" walks AUTO_ORDER; an explicit backend is tried alone. */
function backendOrder(cfg: RelevanceGateConfig): GateBackendImpl[] {
	const keys = cfg.backend === "auto" ? AUTO_ORDER : [cfg.backend];
	return keys.map((k) => GATE_BACKENDS[k]).filter((b): b is GateBackendImpl => !!b);
}

// ---------------------------------------------------------------- gate stats (for /annotations-gate status)

export interface GateStats {
	backend: string;
	judged: number;
	hidden: number;
	hiddenStale: number;
	latencyMs: number;
	at: string;
	/** Bounded snapshot of what was sent to the deciding backend on the last judged call. */
	request?: GateDebugRequest;
	/** Verdicts plus raw backend reply from the last judged call. */
	response?: GateDebugResponse;
}

/** Trimmed copy of a GateRequest for status display (caps keep memory and notifications small). */
export interface GateDebugRequest {
	task: string;
	tool: string;
	path?: string;
	candidates: { id: string; path: string; loc: string; stale: boolean; text: string }[];
}

export interface GateDebugResponse {
	/** One keep/drop per request candidate, in request order. */
	verdicts: boolean[];
	/** Raw backend reply (chat text, classifier answers) when the backend provides it. */
	raw?: unknown;
}

const DEBUG_TASK_CAP = 600;
const DEBUG_CAND_TEXT_CAP = 240;
const DEBUG_RAW_CAP = 2000;

function snapshotRequest(req: GateRequest): GateDebugRequest {
	const cut = (s: string, cap: number) => (s.length > cap ? `${s.slice(0, cap)}… [${s.length} chars]` : s);
	return {
		task: cut(req.task, DEBUG_TASK_CAP),
		tool: req.tool,
		path: req.path,
		candidates: req.candidates.map((c) => ({
			id: c.id,
			path: c.path,
			loc: c.loc,
			stale: c.stale,
			text: cut(c.text, DEBUG_CAND_TEXT_CAP),
		})),
	};
}

/** Render the last gate call for /annotations-gate status: outcome, plus what was sent and what came back. */
export function formatLastGateCall(stats: GateStats | null): string {
	if (!stats) return "last gate call: none yet";
	const head = `last gate call: ${stats.backend}, judged ${stats.judged}, hidden ${stats.hidden}${stats.hiddenStale ? ` (${stats.hiddenStale} stale)` : ""}, ${stats.latencyMs}ms at ${stats.at}`;
	if (!stats.request && !stats.response) return head;
	const lines = [head];
	if (stats.request) {
		const req = stats.request;
		lines.push(`  sent: tool=${req.tool}${req.path ? ` path=${req.path}` : ""}`);
		lines.push(`    task: ${JSON.stringify(req.task)}`);
		lines.push(`    candidates (${req.candidates.length}):`);
		req.candidates.forEach((c, i) => {
			lines.push(`      #${i} ${c.id.slice(0, 8)} ${c.path} ${c.loc}${c.stale ? " [stale]" : ""}: ${JSON.stringify(c.text)}`);
		});
	}
	if (stats.response) {
		lines.push(`  reply: keep=[${stats.response.verdicts.map((v) => (v ? "T" : "F")).join(" ")}]`);
		if (stats.response.raw !== undefined) {
			let raw: string;
			try {
				raw = JSON.stringify(stats.response.raw) ?? String(stats.response.raw);
			} catch {
				raw = String(stats.response.raw);
			}
			if (raw.length > DEBUG_RAW_CAP) raw = `${raw.slice(0, DEBUG_RAW_CAP)}… [${raw.length} chars]`;
			lines.push(`    raw: ${raw}`);
		}
	}
	return lines.join("\n");
}

let lastStats: GateStats | null = null;
export function getLastGateStats(): GateStats | null {
	return lastStats;
}

// ---------------------------------------------------------------- verdict cache

const verdictCache = new Map<string, boolean>();

function verdictKey(task: string, c: GateCandidate): string {
	return sha256(`${task}\n${c.id}\n${c.text}\n${c.loc}\n${c.stale ? "stale" : "fresh"}`);
}

// ---------------------------------------------------------------- orchestration

function withTimeout<T>(ms: number, parent: AbortSignal | undefined, fn: (signal: AbortSignal) => Promise<T>): Promise<T> {
	const ctrl = new AbortController();
	const timer = setTimeout(() => ctrl.abort(new Error(`gate timeout after ${ms}ms`)), ms);
	const onParent = () => ctrl.abort(parent?.reason ?? new Error("aborted"));
	if (parent) parent.addEventListener("abort", onParent, { once: true });
	return fn(ctrl.signal).finally(() => {
		clearTimeout(timer);
		if (parent) parent.removeEventListener("abort", onParent);
	});
}

function passthrough(items: GateCandidate[], backend: string, latencyMs = 0): GateOutcome {
	lastStats = { backend, judged: 0, hidden: 0, hiddenStale: 0, latencyMs, at: new Date().toISOString() };
	return { items, hidden: 0, hiddenStale: 0, backend, latencyMs };
}

/**
 * Filter candidate annotations by relevance to the current task. Never throws:
 * every failure mode returns all items (fail-open). When items are hidden, the
 * caller appends a trace line so the model knows hidden notes exist.
 */
export async function applyRelevanceGate(
	ctx: ExtensionContext,
	repoRoot: string,
	tool: string,
	path: string | undefined,
	items: GateCandidate[],
): Promise<GateOutcome> {
	const started = Date.now();
	try {
		const cfg = (await loadConfig(repoRoot)).relevanceGate;
		if (!cfg.enabled || cfg.backend === "off") return passthrough(items, "skipped");
		if (items.length < cfg.minItemsToGate) return passthrough(items, "skipped");

		const task = await getTaskText(ctx, cfg.taskMaxChars);
		if (!task) return passthrough(items, "skipped:no-task");

		// Cache hit?
		const keep: boolean[] = new Array(items.length).fill(true);
		const todo: number[] = [];
		items.forEach((c, i) => {
			const hit = verdictCache.get(verdictKey(task, c));
			if (hit === undefined) todo.push(i);
			else keep[i] = hit;
		});

		// Cap the judged set; overflow is kept unjudged (fail-open).
		const judged = todo.slice(0, cfg.maxCandidates);
		for (const i of todo.slice(cfg.maxCandidates)) keep[i] = true;

		let debugReq: GateDebugRequest | undefined;
		let lastResponse: GateDebugResponse | undefined;
		if (judged.length > 0) {
			const req: GateRequest = { task, tool, path, candidates: judged.map((i) => items[i]) };
			debugReq = snapshotRequest(req);
			let verdicts: boolean[] | null = null;
			let rawReply: unknown;
			let hasRaw = false;
			let usedBackend = "";
			for (const backend of backendOrder(cfg)) {
				try {
					if (!(await backend.available(ctx, cfg))) continue;
					const v = await withTimeout(cfg.timeoutMs, ctx.signal, (signal) => backend.judge(ctx, cfg, req, signal));
					const keepArr = Array.isArray(v) ? v : v?.keep;
					if (!Array.isArray(keepArr) || keepArr.length !== judged.length) {
						debug(`backend ${backend.name} returned ${keepArr?.length ?? "non-array"} verdicts for ${judged.length} candidates — trying next`);
						continue;
					}
					verdicts = keepArr;
					if (!Array.isArray(v) && v && "raw" in v) {
						rawReply = v.raw;
						hasRaw = true;
					}
					usedBackend = backend.name;
					break;
				} catch (err: any) {
					debug(`backend ${backend.name} failed — trying next: ${err?.message ?? err}`);
				}
			}
			if (!verdicts) {
				debug("no gate backend succeeded — fail-open");
				return passthrough(items, "fail-open", Date.now() - started);
			}
			judged.forEach((itemIdx, j) => {
				keep[itemIdx] = verdicts![j] !== false; // undefined/true → keep; only explicit false drops
				verdictCache.set(verdictKey(task, items[itemIdx]), keep[itemIdx]);
			});
			debug(`${usedBackend}: kept ${verdicts.filter(Boolean).length}/${judged.length} judged, ${todo.length - judged.length} overflow kept`);
			if (hasRaw) {
				try {
					debug(`raw reply: ${JSON.stringify(rawReply)?.slice(0, 500)}`);
				} catch {
					debug("raw reply: [unserializable]");
				}
			}
			lastResponse = { verdicts: verdicts.slice(), ...(hasRaw ? { raw: rawReply } : {}) };
		}

		const kept = items.filter((_, i) => keep[i]);
		const hidden = items.length - kept.length;
		const hiddenStale = items.filter((c, i) => !keep[i] && c.stale).length;
		lastStats = { backend: hidden === 0 ? "kept-all" : "gated", judged: todo.length, hidden, hiddenStale, latencyMs: Date.now() - started, at: new Date().toISOString(), request: debugReq, response: lastResponse };
		return { items: kept, hidden, hiddenStale, backend: lastStats.backend, latencyMs: lastStats.latencyMs };
	} catch (err: any) {
		debug(`gate error — fail-open: ${err?.message ?? err}`);
		return passthrough(items, "fail-open:error", Date.now() - started);
	}
}

/** Trace line appended when the gate hid notes, so the model can pull them via `annotate list`. */
export function hiddenTrace(hidden: number, hiddenStale: number): string {
	const staleNote = hiddenStale > 0 ? ` (${hiddenStale} stale)` : "";
	return `\n─── annotations: ${hidden} hidden as not relevant to the current task${staleNote} — \`annotate list\` to see all ───`;
}
