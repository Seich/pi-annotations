/**
 * Relevance-gate backends.
 *
 * All backends implement GateBackendImpl (see relevance.ts): one batched judge()
 * call per tool result, returning one keep/drop boolean per candidate note.
 *
 *  - "python": a local decision model (Julia-1 or Laya) run by gate-sidecar.py in
 *    a dedicated venv, talked to over stdio JSON-RPC. Registered as a pi provider
 *    with a custom classifier API, so ctx.modelRegistry.classify() works on it.
 *    Installed explicitly via `/annotations-gate setup` — nothing is downloaded
 *    or installed without consent. Weights are pinned to immutable revisions.
 *  - "llama.cpp": the user's own llama.cpp router; its models are already listed
 *    as classifiers by pi. No spawn, no install.
 *  - "api": hosted classifier models (Jev/Clef/...) via working credentials.
 *  - "chat": the session chat model answering a JSON keep-list (zero setup).
 *
 * "auto" tries them in the order AUTO_ORDER and uses the first that works.
 * Classifier backends apply keepThreshold/ambiguityBand to the bool probability:
 * keep when P(relevant) >= threshold, keep when |P - 0.5| < band (fail-open on
 * ambiguity — small decision models are miscalibrated, so unsure means keep).
 *
 * Debug: PI_ANNOTATIONS_DEBUG=1.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { completeSimple } from "@earendil-works/pi-ai/compat";
import type { Api, AssistantMessage, ClassifierApi, ClassifierContext, ClassifierModel, ClassifierQuestion, ClassifierResult, Context, Model } from "@earendil-works/pi-ai";
import type { RelevanceGateConfig } from "./config.ts";
import type { GateBackendImpl, GateCandidate, GateRequest } from "./relevance.ts";

const DEBUG = !!process.env.PI_ANNOTATIONS_DEBUG;
function debug(...args: unknown[]): void {
	if (DEBUG) console.error("[pi-annotations:gate]", ...args);
}

// ---------------------------------------------------------------- paths & specs

const SRC_DIR = dirname(fileURLToPath(import.meta.url));
export const EXT_DIR = dirname(SRC_DIR);
export const SIDECAR_PATH = join(SRC_DIR, "gate-sidecar.py");
export const STATE_DIR = join(EXT_DIR, ".gate");
const VENV_DIR = join(STATE_DIR, "venv");
const SETUP_FILE = join(STATE_DIR, "setup.json");

export const LOCAL_PROVIDER = "pi-annotations-local";
const LOCAL_API = "pi-annotations-gate" as ClassifierApi;

export interface EngineSpec {
	engine: string;
	label: string;
	license: string;
	approxSize: string;
	/** Minimum python version (julia-1's runtime requires >= 3.11). */
	minPython: [number, number];
	/** pip arg arrays run in the venv BEFORE the weights snapshot (torch CPU wheel etc.). */
	pipPre: string[][];
	/** pip arg arrays run AFTER the snapshot; "{SNAPSHOT_DIR}" is substituted with it. */
	pipPost: string[][];
	/** HF repo whose HEAD commit pins everything (resolved via the HF API at setup, recorded in setup.json). */
	weightsRepo: string;
	/** Subdir of STATE_DIR the HF snapshot goes to; undefined = the package self-downloads weights (laya, pinned via LAYA_REVISION). */
	weightsDir?: string;
}

const TORCH_CPU = ["install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"];

export const ENGINES: Record<string, EngineSpec> = {
	"julia-1": {
		engine: "julia-1",
		label: "Julia-1 — 144M multilingual decision model (Supersonic Labs)",
		license: "Apache-2.0",
		approxSize: "~0.6 GB snapshot + ~1 GB venv (torch CPU)",
		minPython: [3, 11],
		// Everything comes from HuggingFace over plain HTTPS — the HF repo ships both the
		// weights AND the python package (pyproject.toml, package "supersonic-julia").
		// No git, no GitHub: a git+https clone can block on a credential prompt when the
		// server answers 401/404 with a login dialog (that hang is why this is HF-only).
		pipPre: [TORCH_CPU],
		pipPost: [["install", "{SNAPSHOT_DIR}"]],
		weightsRepo: "SupersonicLabs/Julia-1",
		weightsDir: "weights/julia-1",
	},
	laya: {
		engine: "laya",
		// typed-decisions checkpoint (the base checkpoint is near-chance on typed questions);
		// measured on this repo: clean relevant/irrelevant separation where julia-1 (144M)
		// gave confident-but-wrong answers. ~0.6s per predict on CPU after warmup.
		label: "Laya typed-decisions — 421M decision model (Convai Innovations)",
		license: "Apache-2.0",
		approxSize: "~0.8 GB weights + ~1 GB venv (torch CPU)",
		minPython: [3, 10],
		pipPre: [TORCH_CPU, ["install", "laya==0.3.27"]],
		pipPost: [],
		weightsRepo: "convaiinnovations/laya",
	},
};
export const DEFAULT_ENGINE = "laya";

export interface SetupInfo {
	engine: string;
	python: string;
	pipPre: string[][];
	pipPost: string[][];
	weightsRepo: string;
	/** HF commit sha everything was pinned to at setup time. */
	snapshotRevision: string;
	createdAt: string;
}

export async function readSetupInfo(): Promise<SetupInfo | null> {
	try {
		return JSON.parse(await readFile(SETUP_FILE, "utf-8"));
	} catch {
		return null;
	}
}

function venvPython(): string {
	return join(VENV_DIR, process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
}

// ---------------------------------------------------------------- sidecar process

export class Sidecar {
	private proc: ChildProcess | null = null;
	private nextId = 1;
	private pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();
	private buf = "";
	private readyResolve!: () => void;
	private readyReject!: (e: any) => void;
	readonly ready: Promise<void>;
	engine = "";

	constructor() {
		this.ready = new Promise((res, rej) => {
			this.readyResolve = res;
			this.readyReject = rej;
		});
	}

	get running(): boolean {
		return !!this.proc;
	}

	async start(pythonBin: string, engine: string, modelDir: string | undefined, env: Record<string, string>, readyTimeoutMs = 180_000): Promise<void> {
		if (this.proc) return;
		const args = [SIDECAR_PATH, "--engine", engine];
		if (modelDir) args.push("--model-dir", modelDir);
		this.proc = spawn(pythonBin, args, { stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, HF_HUB_DISABLE_PROGRESS_BARS: "1", ...env } });
		// Forward only real error lines: drop progress bars (╸/it/s/─), Fetching, and warnings.
		let errBuf = "";
		this.proc.stderr?.on("data", (d) => {
			errBuf += d.toString("utf8");
			let i: number;
			while ((i = errBuf.indexOf("\n")) >= 0) {
				const line = errBuf.slice(0, i).replace(/\r/g, "\n");
				errBuf = errBuf.slice(i + 1);
				for (const l of line.split("\n")) {
					const t = l.trim();
					if (!t) continue;
					if (/it\/s\]|─|Fetching|Warning|warn/i.test(t)) continue;
					debug("sidecar stderr:", t.slice(0, 500));
				}
			}
		});
		this.proc.stdout?.on("data", (d) => this.onData(d));
		this.proc.on("exit", (code) => {
			this.proc = null;
			const err = new Error(`sidecar exited (code ${code})`);
			this.readyReject(err);
			for (const p of this.pending.values()) p.reject(err);
			this.pending.clear();
		});
		const timer = setTimeout(() => {
			this.kill();
			this.readyReject(new Error(`sidecar ready timeout (${readyTimeoutMs}ms)`));
		}, readyTimeoutMs);
		try {
			await this.ready;
		} finally {
			clearTimeout(timer);
		}
	}

	private onData(d: Buffer): void {
		this.buf += d.toString("utf8");
		let i: number;
		while ((i = this.buf.indexOf("\n")) >= 0) {
			const line = this.buf.slice(0, i);
			this.buf = this.buf.slice(i + 1);
			this.onLine(line);
		}
	}

	private onLine(line: string): void {
		if (!line.trim()) return;
		let msg: any;
		try {
			msg = JSON.parse(line);
		} catch {
			return;
		}
		if (msg.ready) {
			this.engine = msg.engine ?? "";
			this.readyResolve();
			return;
		}
		if (msg.id !== undefined && this.pending.has(msg.id)) {
			const p = this.pending.get(msg.id)!;
			this.pending.delete(msg.id);
			if (msg.error) p.reject(new Error(String(msg.error).slice(0, 500)));
			else p.resolve(msg);
		}
	}

	private rpc(req: Record<string, unknown>, timeoutMs: number, signal?: AbortSignal): Promise<any> {
		if (!this.proc?.stdin?.writable) return Promise.reject(new Error("sidecar not running"));
		const id = this.nextId++;
		const proc = this.proc;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`sidecar rpc timeout (${timeoutMs}ms)`));
			}, timeoutMs);
			const onAbort = () => {
				clearTimeout(timer);
				this.pending.delete(id);
				reject(new Error("aborted"));
			};
			signal?.addEventListener("abort", onAbort, { once: true });
			this.pending.set(id, {
				resolve: (v) => {
					clearTimeout(timer);
					signal?.removeEventListener("abort", onAbort);
					resolve(v);
				},
				reject: (e) => {
					clearTimeout(timer);
					signal?.removeEventListener("abort", onAbort);
					reject(e);
				},
			});
			proc.stdin!.write(JSON.stringify({ id, ...req }) + "\n");
		});
	}

	async ping(timeoutMs = 10_000, signal?: AbortSignal): Promise<boolean> {
		const r = await this.rpc({ ping: true }, timeoutMs, signal);
		return r.pong === true;
	}

	async classify(context: ClassifierContext, signal?: AbortSignal): Promise<ClassifierResult> {
		const r = await this.rpc({ state: context.state, questions: context.questions }, 120_000, signal);
		return { api: LOCAL_API, provider: LOCAL_PROVIDER, model: this.engine, answers: r.answers ?? {}, stopReason: "stop", timestamp: Date.now() };
	}

	kill(): void {
		try {
			this.proc?.kill("SIGKILL");
		} catch {
			// already gone
		}
		this.proc = null;
	}
}

let sidecar: Sidecar | null = null;
export function getSidecar(): Sidecar {
	return (sidecar ??= new Sidecar());
}

// ---------------------------------------------------------------- provider registration

let providerRegisteredFor: string | null = null;

/** Register the local sidecar as a pi provider with a custom classifier API (idempotent per engine). */
export function ensureLocalProvider(pi: ExtensionAPI, engine: string): void {
	if (providerRegisteredFor === engine) return;
	pi.registerProvider(LOCAL_PROVIDER, {
		name: "pi-annotations local gate",
		// baseUrl/apiKey are schema-required when defining models, but the custom classifier
		// API below never opens a socket — all traffic is stdio to the sidecar process.
		// The discard port is a placeholder to satisfy validation.
		baseUrl: "http://127.0.0.1:1",
		apiKey: "local",
		authHeader: false,
		models: [
			{
				type: "classifier",
				id: engine,
				name: `${engine} (local sidecar)`,
				api: LOCAL_API,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 8192,
			},
		],
		classifiers: {
			[LOCAL_API]: {
				classify: async (_model: ClassifierModel<ClassifierApi>, context: ClassifierContext, options?: { signal?: AbortSignal }): Promise<ClassifierResult> => {
					try {
						return await sidecarClassify(context, options?.signal);
					} catch (err: any) {
						debug("sidecar classify failed:", err?.message ?? err);
						return { api: LOCAL_API, provider: LOCAL_PROVIDER, model: "sidecar", answers: {}, stopReason: "error", errorMessage: String(err?.message ?? err), timestamp: Date.now() };
					}
				},
			},
		},
	});
	providerRegisteredFor = engine;
}

async function sidecarClassify(context: ClassifierContext, signal?: AbortSignal): Promise<ClassifierResult> {
	const info = await readSetupInfo();
	if (!info) throw new Error("local gate not set up — run /annotations-gate setup");
	const sc = getSidecar();
	if (!sc.running) {
		const spec = ENGINES[info.engine];
		const modelDir = spec?.weightsDir ? join(STATE_DIR, spec.weightsDir) : undefined;
		const env: Record<string, string> = info.engine === "laya" ? { LAYA_REVISION: info.snapshotRevision } : {};
		await sc.start(venvPython(), info.engine, modelDir, env);
	}
	return sc.classify(context, signal);
}

/** Start the sidecar in the background at session start when setup exists (warm model). */
export async function warmLocalGate(pi: ExtensionAPI): Promise<void> {
	const info = await readSetupInfo();
	if (!info) return;
	ensureLocalProvider(pi, info.engine);
	const sc = getSidecar();
	if (sc.running) return;
	const spec = ENGINES[info.engine];
	const modelDir = spec?.weightsDir ? join(STATE_DIR, spec.weightsDir) : undefined;
	const env: Record<string, string> = info.engine === "laya" ? { LAYA_REVISION: info.snapshotRevision } : {};
	sc.start(venvPython(), info.engine, modelDir, env).catch((err) => debug("warm start failed:", err?.message ?? err));
}

// ---------------------------------------------------------------- setup & status

export type ExecFn = (
	cmd: string,
	args: string[],
	opts?: { cwd?: string; timeout?: number; signal?: AbortSignal },
) => Promise<{ code: number; stdout: string; stderr: string }>;

/**
 * Resolve an HF model repo's HEAD commit via the REST API (`/api/models/<repo>` → .sha)
 * over plain HTTPS with curl -sf. Never via `git ls-remote`/`git clone`: git blocks on an
 * interactive credential prompt when the server answers 401/404 with a login dialog
 * (on macOS the osxkeychain helper pops a GUI). curl fails fast instead.
 */
async function resolveHfRevision(exec: ExecFn, repo: string, signal?: AbortSignal): Promise<string | undefined> {
	const r = await exec("curl", ["-sfL", "--max-time", "30", `https://huggingface.co/api/models/${repo}`], { timeout: 45_000, signal });
	if (r.code !== 0) return undefined;
	try {
		return JSON.parse(r.stdout)?.sha ?? undefined;
	} catch {
		return undefined;
	}
}

async function findPython(exec: ExecFn, min: [number, number], signal?: AbortSignal): Promise<string> {
	for (const cand of ["python3", "python"]) {
		const r = await exec(cand, ["--version"], { timeout: 15_000, signal });
		if (r.code !== 0) continue;
		const m = r.stdout.match(/Python\s+(\d+)\.(\d+)/);
		if (m && (Number(m[1]) > min[0] || (Number(m[1]) === min[0] && Number(m[2]) >= min[1]))) return cand;
	}
	throw new Error(`python >= ${min[0]}.${min[1]} not found — the local gate backend needs it`);
}

/**
 * Explicit setup: create the venv, install pinned packages, pin + download
 * weights, smoke-test a classify call. Nothing here runs without the user
 * invoking `/annotations-gate setup` and confirming the plan.
 */
export async function setupLocalBackend(
	engine: string,
	exec: ExecFn,
	notify: (message: string, type?: "info" | "warning" | "error") => void,
): Promise<void> {
	const spec = ENGINES[engine];
	if (!spec) throw new Error(`unknown engine ${JSON.stringify(engine)} — expected one of: ${Object.keys(ENGINES).join(", ")}`);

	// Hard overall cap so setup can never hang silently (each step also has its own timeout).
	const setupAbort = new AbortController();
	const overallTimer = setTimeout(() => setupAbort.abort(new Error("setup overall timeout (20 min)")), 20 * 60_000);
	const sig = setupAbort.signal;
	let python = "";
	let snapshotRevision = "";
	let pipPre: string[][] = spec.pipPre;
	const snapshotDir = spec.weightsDir ? join(STATE_DIR, spec.weightsDir) : undefined;
	try {
		notify(`Checking python…`, "info");
		python = await findPython(exec, spec.minPython, sig);

		// Pin the immutable HF revision FIRST — nothing installs or downloads unpinned.
		// Resolved via the HF REST API with curl (never git ls-remote: git blocks on a
		// credential prompt when the server answers 401/404 with a login dialog).
		snapshotRevision = (await resolveHfRevision(exec, spec.weightsRepo, sig)) ?? "";
		if (!snapshotRevision) throw new Error(`could not resolve huggingface.co/${spec.weightsRepo} HEAD — refusing to install unpinned`);
		debug("pinned HF revision:", snapshotRevision);

		notify(`Creating venv at ${VENV_DIR}…`, "info");
		await mkdir(STATE_DIR, { recursive: true });
		let r = await exec(python, ["-m", "venv", VENV_DIR], { timeout: 120_000, signal: sig });
		if (r.code !== 0) throw new Error(`venv creation failed: ${r.stderr}`);

		const pip = join(VENV_DIR, process.platform === "win32" ? "Scripts/pip.exe" : "bin/pip");
		const py = venvPython();

		// snapshot_download runs with the venv python BEFORE the engine package (and its
		// huggingface_hub dep) is installed — so huggingface_hub must go in first.
		pipPre = spec.weightsDir ? [["install", "huggingface_hub"], ...spec.pipPre] : spec.pipPre;
		for (const args of pipPre) {
			notify(`pip ${args.join(" ")}…`, "info");
			r = await exec(pip, args, { timeout: 900_000, signal: sig });
			if (r.code !== 0) throw new Error(`pip ${args.join(" ")} failed: ${r.stderr.slice(0, 2000)}`);
		}

		if (snapshotDir) {
			notify(`Downloading ${spec.weightsRepo}@${snapshotRevision.slice(0, 12)} → ${snapshotDir}…`, "info");
			const script = `from huggingface_hub import snapshot_download; snapshot_download(${JSON.stringify(spec.weightsRepo)}, revision=${JSON.stringify(snapshotRevision)}, local_dir=${JSON.stringify(snapshotDir)})`;
			r = await exec(py, ["-c", script], { timeout: 1_800_000, signal: sig });
			if (r.code !== 0) throw new Error(`snapshot download failed: ${r.stderr.slice(0, 2000)}`);
		}

		for (const args of spec.pipPost) {
			const resolved = args.map((a) => a.replaceAll("{SNAPSHOT_DIR}", snapshotDir ?? ""));
			notify(`pip ${resolved.join(" ")}…`, "info");
			r = await exec(pip, resolved, { timeout: 900_000, signal: sig });
			if (r.code !== 0) throw new Error(`pip ${resolved.join(" ")} failed: ${r.stderr.slice(0, 2000)}`);
		}
	} finally {
		clearTimeout(overallTimer);
	}

	const info: SetupInfo = { engine, python, pipPre, pipPost: spec.pipPost, weightsRepo: spec.weightsRepo, snapshotRevision, createdAt: new Date().toISOString() };
	await writeFile(SETUP_FILE, JSON.stringify(info, null, 2) + "\n", "utf-8");

	notify("Smoke-testing the gate model…", "info");
	const sc = new Sidecar();
	try {
		const env: Record<string, string> = engine === "laya" ? { LAYA_REVISION: snapshotRevision } : {};
		await sc.start(venvPython(), engine, snapshotDir, env);
		const ok = await sc.ping();
		const res = await sc.classify(
			{ state: { smoke: "pi-annotations gate smoke test" }, questions: { ok: { type: "bool", instructions: "Is this a smoke test?", criteria: { true: "yes it is", false: "no" } } } },
			undefined,
		);
		if (!ok || res.stopReason !== "stop" || res.answers.ok?.type !== "bool") throw new Error(`smoke classify failed: ${res.errorMessage ?? JSON.stringify(res.answers)}`);
		notify(`Smoke test OK — P(ok)=${(res.answers.ok as any).probability?.toFixed(3)}`, "info");
	} finally {
		sc.kill();
	}
	notify(`Local gate backend ready: ${spec.label} (${spec.license}), pinned ${spec.weightsRepo}@${snapshotRevision.slice(0, 12)}. Enable with "relevanceGate.enabled": true in .pi/annotations/config.json`, "info");
}

/** Human-readable status for `/annotations-gate status`. */
export async function gateStatusText(ctx: ExtensionContext, gateCfg: RelevanceGateConfig): Promise<string> {
	const lines: string[] = [];
	lines.push(`gate enabled: ${gateCfg.enabled}, backend: ${gateCfg.backend}${gateCfg.localModel ? `, localModel: ${gateCfg.localModel}` : ""}${gateCfg.classifierModel ? `, classifierModel: ${gateCfg.classifierModel}` : ""}`);
	const info = await readSetupInfo();
	if (info) {
		lines.push(`python sidecar: set up (${info.engine}, python ${info.python}, ${info.createdAt.slice(0, 10)}, weights ${info.weightsRepo ?? "—"}@${(info.snapshotRevision).slice(0, 12)}) — ${getSidecar().running ? `running (${getSidecar().engine})` : "not running (starts on first gate call / session start)"}`);
	} else {
		lines.push("python sidecar: not set up — run /annotations-gate setup");
	}
	try {
		const llama = await ctx.modelRegistry.getAvailableOfType("classifier", "llama.cpp");
		lines.push(`llama.cpp router: ${llama.length ? `${llama.length} classifier model(s): ${llama.map((m) => m.id).join(", ")}` : "none reachable"}`);
	} catch {
		lines.push("llama.cpp router: none reachable");
	}
	try {
		const hosted = (await ctx.modelRegistry.getAvailableOfType("classifier")).filter((m) => m.provider !== "llama.cpp" && m.provider !== LOCAL_PROVIDER);
		lines.push(`api classifiers: ${hosted.length ? hosted.map((m) => `${m.provider}/${m.id}`).join(", ") : "no credentials"}`);
	} catch {
		lines.push("api classifiers: no credentials");
	}
	lines.push(`chat fallback: ${ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "no model"}`);
	return lines.join("\n");
}

// ---------------------------------------------------------------- classifier judge (shared)

const EST_CHARS_PER_TOKEN = 4;
const CANDIDATE_TEXT_CAP = 400;
const LOCAL_BUDGET_TOKENS = 1024; // julia-1 scored context; laya English is 512 — chunking keeps us under

/**
 * Greedy chunking of candidate indices so each classify call's state+questions
 * stays under budgetTokens. Exported for testing.
 */
export function chunkForBudget(task: string, tool: string, path: string | undefined, candidates: GateCandidate[], budgetTokens: number): number[][] {
	const base = JSON.stringify({ task, tool, path: path ?? null });
	const chunks: number[][] = [];
	let cur: number[] = [];
	let curTokens = Math.ceil(base.length / EST_CHARS_PER_TOKEN) + 16;
	for (let i = 0; i < candidates.length; i++) {
		const c = candidates[i];
		const cand = { path: c.path, loc: c.loc, stale: c.stale, text: c.text.slice(0, CANDIDATE_TEXT_CAP) };
		const q = { type: "bool", instructions: "relevant?", criteria: { true: "relevant", false: "not relevant" } };
		const cost = Math.ceil((JSON.stringify(cand) + JSON.stringify(q)).length / EST_CHARS_PER_TOKEN) + 8;
		if (cur.length > 0 && curTokens + cost > budgetTokens) {
			chunks.push(cur);
			cur = [];
			curTokens = Math.ceil(base.length / EST_CHARS_PER_TOKEN) + 16;
		}
		cur.push(i);
		curTokens += cost;
	}
	if (cur.length) chunks.push(cur);
	return chunks;
}

/**
 * How a classifier backend judges candidates:
 *  - "batched": one classify call per chunk, state = {task, candidates}, one bool question
 *    per candidate. Right for System One APIs (Jev) and llama.cpp chat-classifiers, which
 *    are trained on exactly this batched shape.
 *  - "per-candidate": one classify call per candidate, state = {task, annotation}, a single
 *    choice question with descriptive option labels. Required for the small local encoder
 *    models (julia-1/laya): measured on this repo, a batched state with cross-referencing
 *    instructions ("candidates[2]") does NOT discriminate (all answers collapse to one
 *    label), while per-candidate state + choice with descriptive criteria separates
 *    relevant/irrelevant cleanly (0.99 vs 0.92+). Julia reads options only through their
 *    descriptions — bare "yes"/"no" labels calibrate badly.
 */
type JudgeMode = "batched" | "per-candidate";

/** Task chars included in a per-candidate state (local scored context is ~1k tokens). */
const PER_CANDIDATE_TASK_CAP = 1500;

/** Choice criteria for per-candidate judging — descriptive labels, not bare yes/no. */
const RELEVANCE_CRITERIA = {
	relevant: "the annotation is relevant to the current task",
	irrelevant: "the annotation is not relevant to the current task",
} as Record<string, string>;

/**
 * Map a choice answer to a keep verdict: keep when P(relevant) >= keepThreshold, keep when
 * within ambiguityBand of 0.5 (fail-open on ambiguity — small decision models are
 * miscalibrated). Missing/unreadable answers yield p=0.5 → kept. Exported for testing.
 */
export function keepFromRelevanceChoice(
	answer: { type: string; choice?: string; probabilities?: Record<string, number> } | undefined,
	keepThreshold: number,
	ambiguityBand: number,
): boolean {
	let p = 0.5;
	if (answer && answer.type === "choice") {
		p = typeof answer.probabilities?.relevant === "number" ? answer.probabilities.relevant : answer.choice === "relevant" ? 1 : 0;
	}
	return p >= keepThreshold || Math.abs(p - 0.5) < ambiguityBand;
}

function makeClassifierBackend(
	name: string,
	mode: JudgeMode,
	budgetTokens: number,
	pickModel: (ctx: ExtensionContext, cfg: RelevanceGateConfig) => Promise<ClassifierModel<ClassifierApi> | undefined>,
): GateBackendImpl {
	return {
		name,
		available: async (ctx, cfg) => (await pickModel(ctx, cfg)) !== undefined,
		async judge(ctx, cfg, req, signal) {
			const model = await pickModel(ctx, cfg);
			if (!model) throw new Error(`${name}: no classifier model available`);
			const keep = new Array<boolean>(req.candidates.length).fill(true);

			if (mode === "per-candidate") {
				let next = 0;
				const rawAnswers: unknown[] = new Array(req.candidates.length);
				const worker = async () => {
					while (next < req.candidates.length) {
						const i = next++;
						const c = req.candidates[i];
						// Measured on julia-1: the annotation must be a PLAIN STRING and the state
						// must stay minimal — extra keys (tool/path) or an object-shaped annotation
						// collapse answers to one label. Loc/stale go inline in the string.
						const state = {
							task: req.task.slice(0, PER_CANDIDATE_TASK_CAP),
							annotation: `${c.path} ${c.loc}${c.stale ? " [stale]" : ""}: ${c.text.slice(0, CANDIDATE_TEXT_CAP)}`,
						};
						const questions: Record<string, ClassifierQuestion> = {
							relevant: { type: "choice", instructions: "Is the annotation relevant to the current task?", criteria: RELEVANCE_CRITERIA },
						};
						const result = await ctx.modelRegistry.classify(model, { state, questions }, { signal } as any);
						if (result.stopReason !== "stop") throw new Error(`${name}: classify failed: ${result.errorMessage ?? result.stopReason}`);
						rawAnswers[i] = result.answers.relevant;
						keep[i] = keepFromRelevanceChoice(result.answers.relevant as any, cfg.keepThreshold, cfg.ambiguityBand);
					}
				};
				await Promise.all(Array.from({ length: Math.min(4, req.candidates.length) }, () => worker()));
				debug(`${name}: per-candidate judged ${keep.filter(Boolean).length}/${keep.length} kept`);
				return { keep, raw: rawAnswers };
			}

			const chunks = chunkForBudget(req.task, req.tool, req.path, req.candidates, budgetTokens);
			const rawAnswers: unknown[] = new Array(req.candidates.length);
			for (const chunk of chunks) {
				const state = {
					task: req.task,
					tool: req.tool,
					path: req.path ?? null,
					candidates: chunk.map((i) => {
						const c = req.candidates[i];
						return { path: c.path, loc: c.loc, stale: c.stale, text: c.text.slice(0, CANDIDATE_TEXT_CAP) };
					}),
				};
				const questions: Record<string, ClassifierQuestion> = {};
				chunk.forEach((_, j) => {
					questions[`q${j}`] = {
						type: "bool",
						instructions: `Is candidates[${j}] (in state.candidates) relevant to the current task?`,
						criteria: { true: "relevant to the task", false: "not relevant to the task" },
					};
				});
				const result = await ctx.modelRegistry.classify(model, { state, questions }, { signal } as any);
				if (result.stopReason !== "stop") throw new Error(`${name}: classify failed: ${result.errorMessage ?? result.stopReason}`);
				chunk.forEach((origIdx, j) => {
					const a = result.answers[`q${j}`];
					if (!a || a.type !== "bool") throw new Error(`${name}: missing bool answer q${j}`);
					rawAnswers[origIdx] = a;
					// fail-open on ambiguity: miscalibrated small models → unsure means keep
					keep[origIdx] = a.probability >= cfg.keepThreshold || Math.abs(a.probability - 0.5) < cfg.ambiguityBand;
				});
				debug(`${name}: chunk of ${chunk.length} judged`);
			}
			return { keep, raw: rawAnswers };
		},
	};
}

// ---------------------------------------------------------------- the backends

const localBackend = makeClassifierBackend("python", "per-candidate", LOCAL_BUDGET_TOKENS, async (ctx, cfg) => {
	const info = await readSetupInfo();
	if (!info) return undefined;
	if (cfg.localModel && cfg.localModel !== info.engine) return undefined;
	return ctx.modelRegistry.getModelOfType("classifier", LOCAL_PROVIDER, info.engine) as ClassifierModel<ClassifierApi> | undefined;
});

const llamaCppBackend = makeClassifierBackend("llama.cpp", "batched", 8192, async (ctx, cfg) => {
	const models = await ctx.modelRegistry.getAvailableOfType("classifier", "llama.cpp");
	if (!models.length) return undefined;
	if (cfg.classifierModel) {
		const hit = models.find((m) => m.id === cfg.classifierModel || `llama.cpp/${m.id}` === cfg.classifierModel);
		if (hit) return hit as ClassifierModel<ClassifierApi>;
	}
	return models[0] as ClassifierModel<ClassifierApi>;
});

const apiBackend = makeClassifierBackend("api", "batched", 16_384, async (ctx, cfg) => {
	const models = await ctx.modelRegistry.getAvailableOfType("classifier");
	const hosted = models.filter((m) => m.provider !== "llama.cpp" && m.provider !== LOCAL_PROVIDER);
	if (!hosted.length) return undefined;
	if (cfg.classifierModel) {
		const hit = hosted.find((m) => m.id === cfg.classifierModel || `${m.provider}/${m.id}` === cfg.classifierModel);
		if (hit) return hit as ClassifierModel<ClassifierApi>;
	}
	return (hosted.find((m) => m.id.includes("jev")) ?? hosted[0]) as ClassifierModel<ClassifierApi>;
});

// ---- chat backend (zero-setup fallback): session model answers a JSON keep-list

const GATE_SYSTEM_PROMPT =
	"You are a relevance filter for sidecar code annotations. You receive the current task and a numbered " +
	"list of candidate annotations attached to code the agent just read. Decide which annotations are relevant " +
	"to the task. When unsure, INCLUDE the annotation — a false positive costs a few tokens, a false negative " +
	"loses context. Reply with ONLY a JSON object, no prose.";

/** Build the gate prompt for the chat backend. Exported for testing. */
export function buildGatePrompt(req: GateRequest): string {
	const lines = req.candidates.map((c, i) => `${i}: ${c.path} ${c.loc}${c.stale ? " [stale]" : ""}: ${c.text.replace(/\n/g, " ")}`);
	return [
		`Current task:`,
		req.task || "(unknown)",
		``,
		`Tool result: ${req.tool}${req.path ? ` ${req.path}` : ""}`,
		`Candidate annotations:`,
		...lines,
		``,
		`Reply with ONLY: {"keep": [<indices of relevant candidates>]}. Include any candidate you are unsure about.`,
	].join("\n");
}

/** Parse the chat model's reply into per-candidate keep flags. Throws on anything unparseable (caller fails open). */
export function parseKeepJson(text: string, n: number): boolean[] {
	const m = text.match(/\{[\s\S]*\}/);
	if (!m) throw new Error("no JSON object in gate reply");
	const obj = JSON.parse(m[0]);
	if (!obj || !Array.isArray(obj.keep)) throw new Error("gate reply has no keep array");
	const keep = new Array<boolean>(n).fill(false);
	for (const v of obj.keep) {
		if (typeof v === "number" && Number.isInteger(v) && v >= 0 && v < n) keep[v] = true;
	}
	return keep;
}

function assistantText(msg: AssistantMessage): string {
	return msg.content
		.filter((c: any) => c && c.type === "text" && typeof c.text === "string")
		.map((c: any) => c.text as string)
		.join("\n");
}

function resolveChatModel(ctx: ExtensionContext, cfg: RelevanceGateConfig): Model<Api> | undefined {
	if (cfg.model) {
		const slash = cfg.model.indexOf("/");
		if (slash > 0) {
			const found = ctx.modelRegistry.getModelOfType("chat", cfg.model.slice(0, slash), cfg.model.slice(slash + 1));
			if (found) return found as Model<Api>;
			debug(`configured gate model ${cfg.model} not found, falling back to session model`);
		}
	}
	return ctx.model as Model<Api> | undefined;
}

export const chatBackend: GateBackendImpl = {
	name: "chat",
	available: (ctx) => !!ctx.model,
	async judge(ctx, cfg, req, signal) {
		const model = resolveChatModel(ctx, cfg);
		if (!model) throw new Error("no model available for gate");
		const context: Context = {
			systemPrompt: GATE_SYSTEM_PROMPT,
			messages: [{ role: "user", content: buildGatePrompt(req), timestamp: Date.now() }],
		};
		const msg = await completeSimple(model, context, { maxTokens: 512, temperature: 0, signal, maxRetries: 1 });
		if (msg.stopReason === "error" || msg.stopReason === "aborted") {
			throw new Error(`gate call failed: ${msg.errorMessage ?? msg.stopReason}`);
		}
		const reply = assistantText(msg);
		return { keep: parseKeepJson(reply, req.candidates.length), raw: reply };
	},
};

export const GATE_BACKENDS: Record<string, GateBackendImpl> = {
	python: localBackend,
	"llama.cpp": llamaCppBackend,
	api: apiBackend,
	chat: chatBackend,
};

/** "auto" order: local first (cheapest, private), then router, then hosted API, then chat. */
export const AUTO_ORDER = ["python", "llama.cpp", "api", "chat"];
