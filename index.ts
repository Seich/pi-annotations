/**
 * pi-annotations
 *
 * Sidecar annotations for repos: the agent attaches notes to files, symbols, or line
 * ranges via the `annotate` tool; notes are stored in <repoRoot>/.pi/annotations/ and
 * injected into `read`/`grep` results automatically (always-on context).
 *
 * Human commands:
 *   /annotations [path]        — review / prune notes (delete, edit, re-anchor, archive)
 *   /annotations-view <path>  — visualize a file with its annotations inline
 *   /annotations-export [path] — export one file, or the whole repo, as syntax-highlighted HTML
 *
 * Design rationale lives in this repo's annotations (browse with /annotations);
 * usage is documented in README.md.
 */

import type { TextContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { Type } from "typebox";
import { ScrollView, Text } from "@earendil-works/pi-tui";
import { renderAnnotatedHtml, renderRepoHtml, type RepoFile } from "./src/export-html.ts";
import {
	ANNOTATIONS_DIR,
	deleteNoteFile,
	describeLocation,
	formatAnnotationBlock,
	loadAllNotes,
	loadNote,
	newNoteId,
	rangeHash,
	readLines,
	resolveNotes,
	saveNote,
	toPosix,
	windowHash,
	type Anchor,
	type Note,
	type ResolvedNote,
} from "./src/notes.ts";
import { applyRelevanceGate, formatLastGateCall, getLastGateStats, hiddenTrace, type GateCandidate } from "./src/relevance.ts";
import {
	DEFAULT_ENGINE,
	ENGINES,
	EXT_DIR,
	ensureLocalProvider,
	gateStatusText,
	getSidecar,
	readSetupInfo,
	setupLocalBackend,
	warmLocalGate,
} from "./src/backends.ts";
import { loadConfig, type RelevanceGateConfig } from "./src/config.ts";
import { editGateConfig } from "./src/gate-editor.ts";

const anchorSchema = Type.Union(
	[
		Type.Object({ type: Type.Literal("file") }, { description: "Whole-file annotation" }),
		Type.Object(
			{
				type: Type.Literal("symbol"),
				pattern: Type.String({ description: "Regex matched against file lines to locate the symbol" }),
				occurrence: Type.Optional(Type.Number({ description: "Which match to use when the pattern matches multiple lines (default 0)" })),
			},
			{ description: "Symbol-level annotation (re-resolved on every read)" },
		),
		Type.Object(
			{
				type: Type.Literal("lines"),
				startLine: Type.Number({ description: "First line of the range (1-indexed)" }),
				endLine: Type.Number({ description: "Last line of the range (1-indexed, inclusive)" }),
			},
			{ description: "Line-range annotation; the range is content-hashed so it can be re-located after edits" },
		),
	],
	{ description: "Where the annotation attaches" },
);

type AnchorSpec =
	| { type: "file" }
	| { type: "symbol"; pattern: string; occurrence?: number }
	| { type: "lines"; startLine: number; endLine: number };

export default function (pi: ExtensionAPI) {
	let repoRootCache: { cwd: string; root: string } | null = null;

	/**
	 * VCS root markers, checked while walking up from cwd. `.git` may be a plain file
	 * (worktrees, submodules), so existence is enough — no VCS binary is ever invoked.
	 */
	const VCS_MARKERS = [".git", ".hg", ".jj", ".svn", "._darcs"];

	/**
	 * VCS-agnostic repo-root discovery: walk up from cwd to the nearest directory holding
	 * a VCS marker (git/mercurial/jujutsu/svn/darcs) — pure filesystem, so no VCS command
	 * is run and nothing can block on credentials or a missing binary. Falls back to cwd
	 * when no marker exists, so non-VCS directories work too.
	 */
	async function getRepoRoot(cwd: string): Promise<string> {
		if (repoRootCache?.cwd === cwd) return repoRootCache.root;
		let dir = resolve(cwd);
		for (;;) {
			for (const marker of VCS_MARKERS) {
				try {
					await stat(join(dir, marker));
					repoRootCache = { cwd, root: dir };
					return dir;
				} catch {
					// no marker at this level — keep walking up
				}
			}
			const parent = dirname(dir);
			if (parent === dir) break;
			dir = parent;
		}
		repoRootCache = { cwd, root: resolve(cwd) };
		return repoRootCache.root;
	}

	/** Resolve an agent-supplied path to a repo-relative posix path, or null when outside the repo. */
	function repoRel(root: string, cwd: string, p: string): string | null {
		const rel = toPosix(relative(root, resolve(cwd, p)));
		if (rel.startsWith("..") || isAbsolute(rel)) return null;
		return rel;
	}

	function textResult(text: string, details: unknown) {
		return { content: [{ type: "text", text } satisfies TextContent], details };
	}

	async function buildAnchor(spec: AnchorSpec, lines: string[], expectedText?: string): Promise<Anchor> {
		if (spec.type === "file") return { type: "file" };
		if (spec.type === "symbol") {
			const occurrence = spec.occurrence ?? 0;
			const matches = symbolMatchLines(lines, spec.pattern);
			if (matches.length <= occurrence) {
				throw new Error(`symbol pattern ${JSON.stringify(spec.pattern)} has ${matches.length} match(es); occurrence ${occurrence} does not exist`);
			}
			return { type: "symbol", pattern: spec.pattern, occurrence };
		}
		const { startLine, endLine } = spec;
		if (!Number.isInteger(startLine) || !Number.isInteger(endLine) || startLine < 1 || endLine < startLine || endLine > lines.length) {
			throw new Error(`invalid line range ${startLine}-${endLine} (file has ${lines.length} lines)`);
		}
		if (expectedText !== undefined && lines.slice(startLine - 1, endLine).join("\n") !== expectedText) {
			throw new Error("expectedText does not match the current text of the range — re-read the file and retry");
		}
		return { type: "lines", startLine, endLine, hash: windowHash(lines, startLine, endLine), rangeHash: rangeHash(lines, startLine, endLine) };
	}

	// ---------------------------------------------------------------- annotate tool

	pi.registerTool({
		name: "annotate",
		label: "annotate",
		description:
			"Attach a sidecar annotation (note) to a file, symbol, or line range in this repo, shown whenever " +
			"that code is read or grepped — without writing a comment into the source. " +
			"Actions: add (path+anchor+text), update (id + text and/or anchor), delete (id), list (optionally filtered by path). " +
			"Line-range anchors are content-hashed: they shift automatically when code moves, and are flagged stale " +
			"(never auto-deleted) when the code no longer matches.",
		promptSnippet: "Attach sidecar annotations to files/symbols/line ranges (injected into read/grep results)",
		promptGuidelines: [
			"Use annotate for 'why' notes — intent, tradeoffs, gotchas, rejected alternatives — instead of leaving verbose comments in the code; keep code comments for human-facing API docs and security/contract notes.",
			"Write notes for a future agent with no session context: dense, specific, and self-contained, because they are re-injected on every read/grep of that code.",
			"Pick the tightest anchor that survives edits: symbol for named constructs, lines (with expectedText to verify) for precise ranges, file for module-level orientation.",
			"Maintain notes as code changes: update text when rationale changes, re-anchor or delete stale notes instead of trusting them, and delete notes that no longer earn their context cost.",
			"Never copy annotation text into source edits — annotations are context for reading code, not content of the code.",
		],
		parameters: Type.Object({
			action: Type.Union([Type.Literal("add"), Type.Literal("update"), Type.Literal("delete"), Type.Literal("list")]),
			path: Type.Optional(Type.String({ description: "File path (cwd-relative or repo-relative). Required for add; optional filter for list." })),
			id: Type.Optional(Type.String({ description: "Note id returned by annotate add/list (required for update/delete)" })),
			anchor: Type.Optional(anchorSchema),
			text: Type.Optional(Type.String({ description: "Annotation text. Required for add; new text for update." })),
			expectedText: Type.Optional(
				Type.String({ description: "For lines anchors on add/update: exact current text of the range; the call fails if it doesn't match." }),
			),
		}),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const root = await getRepoRoot(ctx.cwd);

			if (params.action === "add") {
				if (!params.path || !params.anchor || params.text === undefined) throw new Error("add requires path, anchor and text");
				const rel = repoRel(root, ctx.cwd, params.path);
				if (!rel) throw new Error(`path ${params.path} is outside the repo`);
				const lines = await readLines(resolve(root, rel));
				if (!lines) throw new Error(`cannot read ${rel}`);
				const anchor = await buildAnchor(params.anchor, lines, params.expectedText);
				const now = new Date().toISOString();
				const note: Note = { id: newNoteId(), path: rel, anchor, text: params.text, createdAt: now, updatedAt: now, stale: false, archived: false };
				await saveNote(root, note);
				const resolved = (await resolveNotes(root, [note]))[0];
				return textResult(`Annotation added: ${note.id} on ${rel} ${describeLocation(resolved)}`, { id: note.id, path: rel, anchor });
			}

			if (params.action === "update") {
				if (!params.id) throw new Error("update requires id");
				const note = await loadNote(root, params.id);
				if (!note) throw new Error(`no annotation with id ${params.id}`);
				if (params.text !== undefined) note.text = params.text;
				if (params.anchor) {
					const lines = await readLines(resolve(root, note.path));
					if (!lines) throw new Error(`cannot read ${note.path}`);
					note.anchor = await buildAnchor(params.anchor, lines, params.expectedText);
					note.stale = false;
				}
				note.updatedAt = new Date().toISOString();
				await saveNote(root, note);
				const resolved = (await resolveNotes(root, [note]))[0];
				return textResult(`Annotation ${note.id} updated (${describeLocation(resolved)} on ${note.path})`, {
					id: note.id,
					path: note.path,
					anchor: note.anchor,
					stale: resolved.stale,
				});
			}

			if (params.action === "delete") {
				if (!params.id) throw new Error("delete requires id");
				const ok = await deleteNoteFile(root, params.id);
				if (!ok) throw new Error(`no annotation with id ${params.id}`);
				return textResult(`Annotation ${params.id} deleted`, { id: params.id });
			}

			// list
			let notes = await loadAllNotes(root);
			if (params.path) {
				const rel = repoRel(root, ctx.cwd, params.path);
				if (rel) notes = notes.filter((n) => n.path === rel);
			}
			const resolved = await resolveNotes(root, notes);
			if (resolved.length === 0) return textResult("No annotations found", { count: 0 });
			const body = resolved.map(
				(r) =>
					`${r.note.id}  ${r.note.path} ${describeLocation(r)}${r.stale ? " [stale]" : ""}${r.note.archived ? " [archived]" : ""}\n    ${r.note.text.replace(/\n/g, "\n    ")}`,
			);
			return textResult(`${resolved.length} annotation(s):\n\n${body.join("\n\n")}`, {
				count: resolved.length,
				notes: resolved.map((r) => ({
					id: r.note.id,
					path: r.note.path,
					location: describeLocation(r),
					stale: r.stale,
					archived: r.note.archived,
					text: r.note.text,
				})),
			});
		},
	});

	// ---------------------------------------------------------------- relevance-gate setup/status

	// Register the local sidecar provider at load when a previous setup exists (async is fine:
	// registrations during/after load take effect immediately).
	void readSetupInfo().then((info) => {
		if (info) ensureLocalProvider(pi, info.engine);
	});

	pi.on("session_start", async () => {
		await warmLocalGate(pi);
	});

	pi.on("session_shutdown", () => {
		getSidecar().kill();
	});

	/**
	 * Explicit, consent-first setup of the local gate backend (shared by the `setup`
	 * subcommand and the interactive editor's post-save side effect). Nothing touches
	 * disk or network before the confirm dialog.
	 */
	async function runGateSetup(engine: string, ctx: ExtensionContext): Promise<void> {
		const spec = ENGINES[engine];
		if (!spec) {
			ctx.ui.notify(`Unknown engine ${JSON.stringify(engine)} — expected: ${Object.keys(ENGINES).join(", ")}`, "error");
			return;
		}
		const ok = await ctx.ui.confirm(
			`Set up local relevance gate (${engine})`,
			[
				`Engine: ${spec.label}`,
				`License: ${spec.license} — weights pinned to an immutable revision at setup time`,
				`Installs into: ${join(EXT_DIR, ".gate")} (venv + weights, ~${spec.approxSize})`,
				`pip: ${[...spec.pipPre, ...spec.pipPost].map((a) => a.join(" ")).join(" ; ")}`,
				`Weights/code: huggingface.co/${spec.weightsRepo} — HEAD commit pinned at setup (HF REST API, no git), recorded in .gate/setup.json`,
				`Then: smoke-test classify call, and enable "relevanceGate.enabled": true in .pi/annotations/config.json`,
			].join("\n"),
		);
		if (!ok) return;
		try {
			await setupLocalBackend(engine, (cmd, a, o) => pi.exec(cmd, a, o), (m, t) => ctx.ui.notify(m, t));
			getSidecar().kill(); // drop a sidecar still running an older engine, if any
			ensureLocalProvider(pi, engine);
			await warmLocalGate(pi);
		} catch (err: any) {
			ctx.ui.notify(`Setup failed: ${err?.message ?? err}`, "error");
		}
	}

	/**
	 * Bare /annotations-gate: interactive editor for .pi/annotations/config.json
	 * (ctrl+s saves, esc discards — see src/gate-editor.ts). On save, apply the side
	 * effects the new settings imply: offer to install the local model when the python
	 * backend is wanted but missing/different (the model-fetching trigger), warm the
	 * sidecar when the gate is on, kill it when the gate is off (frees the model).
	 */
	async function openGateEditor(ctx: ExtensionContext, root: string): Promise<void> {
		const before = (await loadConfig(root)).relevanceGate;
		const res = await editGateConfig(ctx, root);
		if (!res.saved || !res.config) return;
		const after = res.config.relevanceGate;

		const setupInfo = await readSetupInfo();
		const wantsLocal = after.enabled && after.backend !== "off" && (after.backend === "python" || after.backend === "auto");
		const engine = after.localModel ?? setupInfo?.engine ?? DEFAULT_ENGINE;
		if (wantsLocal && (!setupInfo || setupInfo.engine !== engine)) {
			const spec = ENGINES[engine];
			const ok = await ctx.ui.confirm(
				`Install the local gate model (${engine})?`,
				[
					`The gate is enabled with backend "${after.backend}", but no local setup exists for engine "${engine}".`,
					`Engine: ${spec.label}`,
					`License: ${spec.license} — weights pinned to an immutable HF revision at setup`,
					`Installs into: ${join(EXT_DIR, ".gate")} (venv + weights, ~${spec.approxSize})`,
					`Weights: huggingface.co/${spec.weightsRepo} — plain HTTPS, HEAD commit pinned, recorded in .gate/setup.json`,
				].join("\n"),
			);
			if (ok) await runGateSetup(engine, ctx);
		} else if (setupInfo) {
			ensureLocalProvider(pi, setupInfo.engine);
			if (wantsLocal) await warmLocalGate(pi);
		}
		if ((!after.enabled || after.backend === "off") && getSidecar().running) {
			getSidecar().kill();
			ctx.ui.notify("Gate disabled — stopped the local sidecar", "info");
		}

		const changed = (Object.keys(after) as (keyof RelevanceGateConfig)[]).filter((k) => before[k] !== after[k]);
		ctx.ui.notify(
			`Saved ${join(root, toPosix(ANNOTATIONS_DIR), "config.json")}${changed.length ? ` — changed: ${changed.join(", ")}` : " (no changes)"}`,
			"info",
		);
	}

	pi.registerCommand("annotations-gate", {
		description:
			"Relevance gate for injected annotations: bare /annotations-gate opens an interactive settings editor (ctrl+s saves, then applies side effects like model setup), /annotations-gate status shows the backend chain, /annotations-gate setup [julia-1|laya] installs the local model (explicit consent)",
		handler: async (args, ctx) => {
			const parts = args.trim().split(/\s+/).filter(Boolean);
			const sub = parts[0] ?? "";
			const root = await getRepoRoot(ctx.cwd);
			if (sub === "" && ctx.mode === "tui") {
				await openGateEditor(ctx, root);
				return;
			}
			if (sub === "" || sub === "status") {
				const cfg = (await loadConfig(root)).relevanceGate;
				const stats = getLastGateStats();
				ctx.ui.notify(`${await gateStatusText(ctx, cfg)}\n${formatLastGateCall(stats)}`, "info");
				return;
			}
			if (sub === "setup") {
				await runGateSetup(parts[1] ?? DEFAULT_ENGINE, ctx);
				return;
			}
			ctx.ui.notify("Usage: /annotations-gate (interactive editor in tui mode) | status | setup [julia-1|laya]", "warning");
		},
	});

	// ---------------------------------------------------------------- always-on injection

	pi.on("tool_result", async (event, ctx) => {
		try {
			if (event.isError) return;
			// Notes come from the repo: only inject in trusted projects (prompt-injection vector otherwise).
			if (!ctx.isProjectTrusted()) return;
			if (event.toolName === "read") return await injectRead(event, ctx);
			if (event.toolName === "grep") return await injectGrep(event, ctx);
			if (event.toolName === "bash" || event.toolName === "powershell") return await injectShell(event, ctx);
		} catch {
			// never break a tool result because of annotations
		}
		return undefined;
	});

	async function injectRead(event: any, ctx: ExtensionContext) {
		const input = event.input as { path?: unknown; offset?: unknown; limit?: unknown };
		if (typeof input.path !== "string") return;
		const root = await getRepoRoot(ctx.cwd);
		const rel = repoRel(root, ctx.cwd, input.path);
		if (!rel) return;
		const notes = (await loadAllNotes(root)).filter((n) => n.path === rel && !n.archived);
		if (notes.length === 0) return;
		const lines = await readLines(resolve(root, rel));
		const resolved = await resolveNotes(root, notes);
		const offset = typeof input.offset === "number" ? input.offset : 1;
		const limit = typeof input.limit === "number" ? input.limit : undefined;
		let winEnd = limit !== undefined ? offset + limit - 1 : lines?.length ?? Number.MAX_SAFE_INTEGER;
		// The read tool itself truncates (2000 lines / 50KB): scope the window to the lines
		// actually returned, so notes beyond the truncation point wait for the next offset.
		const truncation = (event.details as any)?.truncation;
		if (truncation?.truncated && typeof truncation.outputLines === "number") {
			winEnd = Math.min(winEnd, offset - 1 + truncation.outputLines);
		}
		const items: GateCandidate[] = [];
		for (const r of resolved) {
			if (r.note.anchor.type === "file" || (r.start <= winEnd && r.end >= offset)) {
				items.push({ id: r.note.id, path: rel, loc: describeLocation(r), text: r.note.text, stale: r.stale });
			}
		}
		const out = await gateAndFormat(ctx, root, "read", rel, items);
		if (!out) return;
		return appendTextBlock(event, out);
	}

	async function injectGrep(event: any, ctx: ExtensionContext) {
		const input = event.input as { path?: unknown };
		const root = await getRepoRoot(ctx.cwd);
		const notes = (await loadAllNotes(root)).filter((n) => !n.archived);
		if (notes.length === 0) return;
		const searchPath = typeof input.path === "string" ? input.path : ".";

		// Parse "path:line: text" (matches) and "path-line- text" (context) output lines.
		const text = event.content
			.filter((c: any) => c.type === "text")
			.map((c: any) => c.text as string)
			.join("\n");
		const matchedByNote = new Map<Note, Set<number>>();
		for (const line of text.split("\n")) {
			const m = line.match(/^(.+?):(\d+): /) ?? line.match(/^(.+?)-(\d+)- /);
			if (!m) continue;
			const rel = resolveGrepPath(root, ctx.cwd, searchPath, m[1]);
			if (!rel) continue;
			const ln = Number(m[2]);
			for (const n of notes) {
				if (pathsMatch(rel, n.path)) {
					let set = matchedByNote.get(n);
					if (!set) matchedByNote.set(n, (set = new Set()));
					set.add(ln);
				}
			}
		}
		if (matchedByNote.size === 0) return;

		const resolved = await resolveNotes(root, [...matchedByNote.keys()]);
		const items: GateCandidate[] = [];
		for (const r of resolved) {
			const lines = matchedByNote.get(r.note);
			if (!lines || lines.size === 0) continue;
			const touches = r.note.anchor.type === "file" || [...lines].some((l) => l >= r.start && l <= r.end);
			if (touches) items.push({ id: r.note.id, path: r.note.path, loc: describeLocation(r), text: r.note.text, stale: r.stale });
		}
		const out = await gateAndFormat(ctx, root, "grep", undefined, items);
		if (!out) return;
		return appendTextBlock(event, out);
	}

	/**
	 * Shell (bash/powershell) output is unstructured, so annotations attach only where the
	 * output provably shows lines of an annotated file, and only for the ranges shown:
	 *  (a) grep/rg-style "path:line:" / "path-line-" prefixes → authoritative line numbers;
	 *      path-less "N:text" lines (single-file grep) count only when a mentioned file's
	 *      Nth line equals the printed text, so numbers from unrelated tools never attach;
	 *  (b) raw file content (cat / sed / head / tail / awk) → aligned back to file lines.
	 *      A line counts only in a run of 2+ consecutive output lines mapping to
	 *      consecutive file lines, or as a unique file line, so coincidental matches on
	 *      common short lines (braces, blanks) never qualify. Both path-less attribution
	 *      and alignment run only for annotated files mentioned in the command, bounding
	 *      the cost.
	 */
	async function injectShell(event: any, ctx: ExtensionContext) {
		const input = event.input as { command?: unknown };
		if (typeof input.command !== "string") return;
		const root = await getRepoRoot(ctx.cwd);
		const notes = (await loadAllNotes(root)).filter((n) => !n.archived);
		if (notes.length === 0) return;
		const output = event.content.filter((c: any) => c.type === "text").map((c: any) => c.text as string).join("\n");
		if (!output.trim()) return;
		const outLines = output.split("\n");

		const readByPath = new Map<string, Set<number>>();

		const addRead = (notePath: string, line: number) => {
			let set = readByPath.get(notePath);
			if (!set) readByPath.set(notePath, (set = new Set()));
			set.add(line);
		};

		// Annotated files mentioned in the command; load their lines once.
		const mentioned = new Map<string, string[] | null>();
		for (const token of tokenizeCommand(input.command)) {
			const rel = repoRel(root, ctx.cwd, token) ?? token;
			for (const n of notes) {
				if ((pathsMatch(token, n.path) || pathsMatch(rel, n.path)) && !mentioned.has(n.path)) {
						mentioned.set(n.path, await readLines(resolve(root, n.path)));
				}
			}
		}

		// (a) grep/rg-style output: "path:line: text" (or "path-line- text")
		for (const raw of outLines) {
			const clean = stripAnsi(raw);
			const prefixed = clean.match(/^(.+?):(\d+):(.*)$/) ?? clean.match(/^(.+?)-(\d+)-(.*)$/);
			if (prefixed) {
				for (const n of notes) {
					if (pathsMatch(prefixed[1], n.path) || pathsMatch(repoRel(root, ctx.cwd, prefixed[1]) ?? prefixed[1], n.path)) {
						addRead(n.path, Number(prefixed[2]));
					}
				}
				continue;
			}
			const pathless = clean.match(/^(\d+):(.*)$/) ?? clean.match(/^(\d+)-(.*)$/);
			if (pathless) {
				const num = Number(pathless[1]);
				const text = pathless[2].replace(/\r$/, "");
				for (const [notePath, lines] of mentioned) {
					if (lines && lines[num - 1]?.replace(/\r$/, "") === text) addRead(notePath, num);
				}
			}
		}
		// (b) raw-content alignment (cat / sed / head / tail / awk) for mentioned files
		for (const [notePath, lines] of mentioned) {
			if (!lines) continue;
			for (const l of findReadLines(lines, outLines)) addRead(notePath, l);
		}
		if (readByPath.size === 0) return;

		const hitNotes = notes.filter((n) => readByPath.has(n.path));
		const resolved = await resolveNotes(root, hitNotes);
		const items: GateCandidate[] = [];
		for (const r of resolved) {
			const lines = readByPath.get(r.note.path)!;
			const touches = r.note.anchor.type === "file" || [...lines].some((l) => l >= r.start && l <= r.end);
			if (touches) items.push({ id: r.note.id, path: r.note.path, loc: describeLocation(r), text: r.note.text, stale: r.stale });
		}
		const out = await gateAndFormat(ctx, root, "bash", undefined, items);
		if (!out) return;
		return appendTextBlock(event, out);
	}

	/**
	 * Run the relevance gate (if enabled) over the candidate notes, then format
	 * the block. When the gate hides notes, a trace line is appended so the model
	 * knows hidden annotations exist and can pull them with `annotate list`.
	 */
	async function gateAndFormat(
		ctx: ExtensionContext,
		root: string,
		tool: string,
		path: string | undefined,
		items: GateCandidate[],
	): Promise<string | null> {
		const gate = await applyRelevanceGate(ctx, root, tool, path, items);
		const block = formatAnnotationBlock(gate.items);
		const trace = gate.hidden > 0 ? hiddenTrace(gate.hidden, gate.hiddenStale) : "";
		const out = (block ?? "") + trace;
		return out || null;
	}

	function resolveGrepPath(root: string, cwd: string, searchPath: string, grepPath: string): string | null {
		const candidates = [resolve(cwd, searchPath, grepPath), resolve(cwd, grepPath), resolve(root, grepPath)];
		for (const abs of candidates) {
			const rel = toPosix(relative(root, abs));
			if (!rel.startsWith("..") && !isAbsolute(rel)) return rel;
		}
		return toPosix(grepPath);
	}

	function appendTextBlock(event: any, block: string) {
		const content = [...(event.content as any[]), { type: "text", text: block } satisfies TextContent];
		return { content, structuredContent: event.structuredContent };
	}

	// ---------------------------------------------------------------- /annotations

	pi.registerCommand("annotations", {
		description: "List and manage annotations: review stale notes, delete, edit, re-anchor, archive",
		handler: async (args, ctx) => {
			const root = await getRepoRoot(ctx.cwd);
			const filter = args.trim();
			const reload = async () => {
				let resolved = await resolveNotes(root, await loadAllNotes(root));
				if (filter) resolved = resolved.filter((r) => r.note.path.includes(filter));
				return resolved;
			};
			let resolved = await reload();
			if (resolved.length === 0) {
				ctx.ui.notify(filter ? `No annotations matching "${filter}"` : "No annotations in this repo", "info");
				return;
			}

			while (true) {
				const options = resolved.map(
					(r) => `${r.stale ? "⚠ stale  " : ""}${r.note.archived ? "[archived] " : ""}${r.note.path} ${describeLocation(r)} — ${firstLine(r.note.text)}`,
				);
				options.push("Prune all stale", "Done");
				const choice = await ctx.ui.select("Annotations", options);
				if (!choice || choice === "Done") return;
				if (choice === "Prune all stale") {
					const stale = resolved.filter((r) => r.stale);
					if (stale.length === 0) {
						ctx.ui.notify("No stale annotations", "info");
						continue;
					}
					const ok = await ctx.ui.confirm("Prune stale annotations", `Delete ${stale.length} stale annotation(s)? This cannot be undone.`);
					if (ok) {
						for (const r of stale) await deleteNoteFile(root, r.note.id);
						ctx.ui.notify(`Deleted ${stale.length} stale annotation(s)`, "info");
					}
					resolved = await reload();
					if (resolved.length === 0) return;
					continue;
				}
				const r = resolved[options.indexOf(choice)];
				if (!r) continue;
				await manageNote(root, r, ctx);
				resolved = await reload();
				if (resolved.length === 0) return;
			}
		},
	});

	async function manageNote(root: string, r: ResolvedNote, ctx: ExtensionContext): Promise<void> {
		const note = r.note;
		const actions = [
			"Delete",
			"Edit text",
			"Re-anchor / move (lines, symbol, or file)",
			note.archived ? "Unarchive (inject into context again)" : "Archive (keep on disk, stop injecting)",
			"Back",
		];
		const action = await ctx.ui.select(`Annotation ${note.id} — ${note.path} ${describeLocation(r)}${r.stale ? " [stale]" : ""}`, actions);
		if (!action || action === "Back") return;

		if (action === "Delete") {
			const ok = await ctx.ui.confirm("Delete annotation", `Delete ${note.id} on ${note.path}?`);
			if (ok) {
				await deleteNoteFile(root, note.id);
				ctx.ui.notify(`Deleted ${note.id}`, "info");
			}
			return;
		}
		if (action === "Edit text") {
			const text = await ctx.ui.input("Annotation text", note.text);
			if (text !== undefined && text.trim()) {
				note.text = text;
				note.updatedAt = new Date().toISOString();
				await saveNote(root, note);
				ctx.ui.notify("Updated", "info");
			}
			return;
		}
		if (action.startsWith("Archive") || action.startsWith("Unarchive")) {
			note.archived = action.startsWith("Unarchive");
			note.updatedAt = new Date().toISOString();
			await saveNote(root, note);
			ctx.ui.notify(note.archived ? "Archived" : "Unarchived", "info");
			return;
		}
		// Re-anchor / move
		const kind = await ctx.ui.select("New anchor type", ["Line range", "Symbol (pattern)", "Whole file"]);
		if (!kind) return;
		const lines = await readLines(resolve(root, note.path));
		if (!lines) {
			ctx.ui.notify(`Cannot read ${note.path}`, "error");
			return;
		}
		try {
			if (kind === "Whole file") {
				note.anchor = { type: "file" };
			} else if (kind === "Symbol (pattern)") {
				const pattern = await ctx.ui.input("Symbol pattern (regex matched against file lines)");
				if (!pattern) return;
				const occurrenceStr = await ctx.ui.input("Occurrence (0-based)", "0");
				const occurrence = Number.parseInt(occurrenceStr ?? "0", 10) || 0;
				const matches = symbolMatchLines(lines, pattern);
				if (matches.length <= occurrence) {
					ctx.ui.notify(`Pattern has ${matches.length} match(es); occurrence ${occurrence} does not exist`, "error");
					return;
				}
				note.anchor = { type: "symbol", pattern, occurrence };
			} else {
				const rangeStr = await ctx.ui.input("Line range, e.g. 12-18 (1-indexed, inclusive)");
				if (!rangeStr) return;
				const m = rangeStr.match(/^(\d+)\s*-\s*(\d+)$/);
				if (!m) {
					ctx.ui.notify("Expected a range like 12-18", "error");
					return;
				}
				const startLine = Number(m[1]);
				const endLine = Number(m[2]);
				if (startLine < 1 || endLine < startLine || endLine > lines.length) {
					ctx.ui.notify(`Invalid range (file has ${lines.length} lines)`, "error");
					return;
				}
				note.anchor = { type: "lines", startLine, endLine, hash: windowHash(lines, startLine, endLine), rangeHash: rangeHash(lines, startLine, endLine) };
			}
			note.stale = false;
			note.updatedAt = new Date().toISOString();
			await saveNote(root, note);
			ctx.ui.notify("Re-anchored", "info");
		} catch (err: any) {
			ctx.ui.notify(`Failed: ${err.message}`, "error");
		}
	}

	// ---------------------------------------------------------------- /annotations-view

	pi.registerCommand("annotations-view", {
		description: "Show a file with its annotations interleaved (never modifies the source file)",
		handler: async (args, ctx) => {
			const pathArg = args.trim();
			if (!pathArg) {
				ctx.ui.notify("Usage: /annotations-view <path>", "warning");
				return;
			}
			const root = await getRepoRoot(ctx.cwd);
			const rel = repoRel(root, ctx.cwd, pathArg);
			if (!rel) {
				ctx.ui.notify(`Path ${pathArg} is outside the repo`, "error");
				return;
			}
			const lines = await readLines(resolve(root, rel));
			if (!lines) {
				ctx.ui.notify(`Cannot read ${rel}`, "error");
				return;
			}
			const notes = (await loadAllNotes(root)).filter((n) => n.path === rel);
			const resolved = await resolveNotes(root, notes);
			const rendered = renderAnnotatedFile(rel, lines, resolved);

			if (ctx.mode === "tui") {
				await ctx.ui.custom((tui, _theme, kb, done) => {
					const scroll = new ScrollView(new Text(rendered.join("\n"), 1, 0), { primary: true, scrollbar: "auto" });
					return {
						render: (width: number) => scroll.render(width),
						invalidate: () => scroll.invalidate(),
						handleInput: (data: string) => {
							if (kb.matches(data, "tui.select.cancel")) {
								done(undefined);
								return;
							}
							(scroll as unknown as { handleInput?(d: string): void }).handleInput?.(data);
							tui.requestRender();
						},
					};
				});
			} else {
				const tmp = join(tmpdir(), `pi-annotations-${basename(rel)}.txt`);
				await writeFile(tmp, rendered.join("\n") + "\n", "utf-8");
				ctx.ui.notify(`Annotated view written to ${tmp}`, "info");
			}
		},
	});

	// ---------------------------------------------------------------- /annotations-export

	pi.registerCommand("annotations-export", {
		description: "Export annotated code as syntax-highlighted HTML: one file with <path>, or all annotated files as one document when no path is given",
		handler: async (args, ctx) => {
			const parts = args.trim().split(/\s+/).filter(Boolean);
			const root = await getRepoRoot(ctx.cwd);

			// No path: one shareable document covering the whole repo.
			if (parts.length === 0) {
				const all = await loadAllNotes(root);
				if (all.length === 0) {
					ctx.ui.notify("No annotations in this repo", "info");
					return;
				}
				const resolved = await resolveNotes(root, all);
				const byPath = new Map<string, ResolvedNote[]>();
				for (const r of resolved) {
					const arr = byPath.get(r.note.path) ?? [];
					arr.push(r);
					byPath.set(r.note.path, arr);
				}
				const files: RepoFile[] = [];
				const missing: ResolvedNote[] = [];
				for (const notePath of [...byPath.keys()].sort()) {
					const lines = await readLines(resolve(root, notePath));
					if (!lines) {
						missing.push(...byPath.get(notePath)!);
						continue;
					}
					files.push({ relPath: notePath, code: lines.join("\n"), resolved: byPath.get(notePath)! });
				}
				const html = await renderRepoHtml(basename(root), files, missing, {
					generatedAt: new Date().toISOString(),
					extensionName: "pi-annotations",
				});
				const out = join(ctx.cwd, "annotations.html");
				await writeFile(out, html, "utf-8");
				ctx.ui.notify(
					`Exported ${files.length} annotated file(s), ${resolved.length} annotation(s)${missing.length ? `, ${missing.length} on missing files` : ""} → ${out}`,
					"info",
				);
				return;
			}

			const pathArg = parts[0];
			const rel = repoRel(root, ctx.cwd, pathArg);
			if (!rel) {
				ctx.ui.notify(`Path ${pathArg} is outside the repo`, "error");
				return;
			}
			const lines = await readLines(resolve(root, rel));
			if (!lines) {
				ctx.ui.notify(`Cannot read ${rel}`, "error");
				return;
			}
			const notes = (await loadAllNotes(root)).filter((n) => n.path === rel);
			const resolved = await resolveNotes(root, notes);
			const html = await renderAnnotatedHtml(rel, lines.join("\n"), resolved, {
				generatedAt: new Date().toISOString(),
				extensionName: "pi-annotations",
			});
			const out = parts[1] ? resolve(ctx.cwd, parts[1]) : join(ctx.cwd, `${basename(rel).replace(/\.[^.]+$/, "")}.annotated.html`);
			await writeFile(out, html, "utf-8");
			ctx.ui.notify(`Exported ${rel} (${resolved.length} annotation(s), ${resolved.filter((r) => r.stale).length} stale) → ${out}`, "info");
		},
	});
}

// ---------------------------------------------------------------- helpers

function stripAnsi(s: string): string {
	return s.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\x1b\][^\x07]*(\x07|\x1b\\\\)/g, "");
}

function tokenizeCommand(command: string): string[] {
	return command
		.split(/\s+/)
		.map((t) => t.replace(/^["'`(]+/, "").replace(/["'`;|)&<>]+$/, ""))
		.filter(Boolean);
}

/** Path equality with safe suffix matching: "src/foo.ts" matches "foo.ts", never "xfoo.ts". */
function pathsMatch(a: string, b: string): boolean {
	if (a === b) return true;
	return a.endsWith("/" + b) || b.endsWith("/" + a);
}

/**
 * Map shell output lines back to file lines. A file line counts as read when it appears in
 * a run of 2+ consecutive output lines matching consecutive file lines, or as a single
 * unique file line (length-gated) — coincidental matches on common short lines (braces,
 * blank lines) never qualify, so no note attaches to output that merely resembles code.
 */
function findReadLines(fileLines: string[], outLines: string[]): Set<number> {
	const byText = new Map<string, number[]>();
	fileLines.forEach((text, i) => {
		const key = stripAnsi(text).replace(/\r/g, "");
		const arr = byText.get(key);
		if (arr) arr.push(i + 1);
		else byText.set(key, [i + 1]);
	});
	const fileLineOf: (number | null)[] = outLines.map((raw) => {
		const nums = byText.get(stripAnsi(raw).replace(/\r/g, ""));
		return nums && nums.length === 1 ? nums[0] : null;
	});
	const read = new Set<number>();
	for (let i = 0; i < outLines.length; i++) {
		const fl = fileLineOf[i];
		if (fl !== null && outLines[i].trim().length >= 4) read.add(fl);
	}
	for (let i = 0; i + 1 < fileLineOf.length; i++) {
		const a = fileLineOf[i];
		const b = fileLineOf[i + 1];
		if (a !== null && b === a + 1) {
			read.add(a);
			read.add(b);
		}
	}
	return read;
}

function firstLine(s: string, max = 60): string {
	const line = s.split("\n", 1)[0].trim();
	return line.length > max ? line.slice(0, max - 1) + "…" : line;
}

function symbolMatchLines(lines: string[], pattern: string): number[] {
	let re: RegExp;
	try {
		re = new RegExp(pattern);
	} catch {
		re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
	}
	const out: number[] = [];
	lines.forEach((line, i) => {
		if (re.test(line)) out.push(i + 1);
	});
	return out;
}

function renderAnnotatedFile(rel: string, lines: string[], resolved: ResolvedNote[]): string[] {
	const gutter = String(Math.max(1, lines.length)).length;
	const byEnd = new Map<number, ResolvedNote[]>();
	const fileLevel: ResolvedNote[] = [];
	for (const r of resolved) {
		if (r.note.anchor.type === "file") fileLevel.push(r);
		else {
			const arr = byEnd.get(r.end) ?? [];
			arr.push(r);
			byEnd.set(r.end, arr);
		}
	}
	const out: string[] = [`${rel} — ${resolved.length} annotation(s)${resolved.some((r) => r.stale) ? " (some stale)" : ""}`];
	for (const r of fileLevel) out.push(...noteLines(gutter, r));
	out.push("─".repeat(gutter + 3));
	lines.forEach((line, i) => {
		const n = i + 1;
		out.push(`${String(n).padStart(gutter)} | ${line}`);
		const notes = byEnd.get(n);
		if (notes) for (const r of notes) out.push(...noteLines(gutter, r));
	});
	return out;
}

function noteLines(gutter: number, r: ResolvedNote): string[] {
	const prefix = `${" ".repeat(gutter)} | ▸ `;
	const head = `${describeLocation(r)}${r.stale ? " [stale: anchored code no longer matches]" : ""}${r.note.archived ? " [archived]" : ""}: ${r.note.text}`;
	const width = Math.max(40, 100 - prefix.length);
	const words = head.split(" ");
	const lines: string[] = [];
	let cur = "";
	for (const w of words) {
		if (cur && cur.length + 1 + w.length > width) {
			lines.push(cur);
			cur = w;
		} else {
			cur = cur ? `${cur} ${w}` : w;
		}
	}
	if (cur) lines.push(cur);
	return lines.map((l, i) => (i === 0 ? prefix + l : `${prefix}  ${l}`));
}
