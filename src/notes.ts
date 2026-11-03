/**
 * Core storage + anchoring logic for pi-annotations.
 *
 * Notes live in <repoRoot>/.pi/annotations/<id>.json (one file per note, so any VCS
 * merges them cleanly and they travel with the repo once the user commits them —
 * the extension itself never runs a VCS command).
 */

import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { join, relative, resolve, sep } from "node:path";
import { withFileMutationQueue } from "@earendil-works/pi-coding-agent";

export const ANNOTATIONS_DIR = join(".pi", "annotations");
/** Context lines hashed on each side of a line-range anchor. */
export const CONTEXT_LINES = 2;
/** Files larger than this are only verified at the recorded position, not scanned. */
export const MAX_SCAN_LINES = 200_000;
/** Cap on injected annotation bytes per tool result. */
export const MAX_ANNOTATION_BLOCK_BYTES = 4096;

export type Anchor =
	| { type: "file" }
	| { type: "symbol"; pattern: string; occurrence: number }
	| { type: "lines"; startLine: number; endLine: number; hash: string; rangeHash: string };

export interface Note {
	id: string;
	/** Repo-relative path, posix separators. */
	path: string;
	anchor: Anchor;
	text: string;
	createdAt: string;
	updatedAt: string;
	/** True when the anchor no longer matches the current code. */
	stale: boolean;
	/** Archived notes are kept on disk but never injected into context. */
	archived: boolean;
}

export interface ResolvedNote {
	note: Note;
	/** Current 1-indexed line range the note applies to (0/0 when unresolvable). */
	start: number;
	end: number;
	stale: boolean;
	/** True when resolveNote() mutated the note (shift or stale-flag change). */
	changed: boolean;
}

export function toPosix(p: string): string {
	return p.split(sep).join("/");
}

export function annotationsDir(repoRoot: string): string {
	return join(repoRoot, ANNOTATIONS_DIR);
}

export function newNoteId(): string {
	return `n_${randomUUID()}`;
}

export function isValidNoteId(id: string): boolean {
	return /^n_[0-9a-z-]+$/i.test(id);
}

function sha256(s: string): string {
	return createHash("sha256").update(s).digest("hex");
}

/** Hash of the range plus CONTEXT_LINES of surrounding context on each side. */
export function windowHash(lines: string[], startLine: number, endLine: number): string {
	const from = Math.max(0, startLine - 1 - CONTEXT_LINES);
	const to = Math.min(lines.length, endLine + CONTEXT_LINES);
	return sha256(lines.slice(from, to).join("\n"));
}

/** Hash of the bare range (fallback when the surrounding context changed). */
export function rangeHash(lines: string[], startLine: number, endLine: number): string {
	return sha256(lines.slice(startLine - 1, endLine).join("\n"));
}

export async function readLines(absPath: string): Promise<string[] | null> {
	try {
		const buf = await readFile(absPath);
		return buf.toString("utf-8").replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
	} catch {
		return null;
	}
}

function escapeRegExp(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function symbolMatches(lines: string[], pattern: string): number[] {
	let re: RegExp;
	try {
		re = new RegExp(pattern);
	} catch {
		re = new RegExp(escapeRegExp(pattern));
	}
	const out: number[] = [];
	lines.forEach((line, i) => {
		if (re.test(line)) out.push(i + 1);
	});
	return out;
}

/**
 * Resolve a note against the current file content, shifting line anchors when the
 * content moved and flagging stale when it no longer matches. Mutates `note`.
 */
export function resolveNote(note: Note, lines: string[] | null): ResolvedNote {
	let changed = false;
	const setStale = (stale: boolean) => {
		if (note.stale !== stale) {
			note.stale = stale;
			changed = true;
		}
	};

	if (note.anchor.type === "file") {
		setStale(false);
		return { note, start: 1, end: lines?.length ?? 0, stale: false, changed };
	}

	if (note.anchor.type === "symbol") {
		const a = note.anchor;
		const matches = lines ? symbolMatches(lines, a.pattern) : [];
		const line = matches[a.occurrence] ?? matches[0];
		const stale = line === undefined;
		setStale(stale);
		return { note, start: line ?? 0, end: line ?? 0, stale, changed };
	}

	// lines anchor
	const a = note.anchor;
	if (!lines) {
		setStale(true);
		return { note, start: a.startLine, end: a.endLine, stale: true, changed };
	}
	if (a.startLine >= 1 && a.endLine <= lines.length && windowHash(lines, a.startLine, a.endLine) === a.hash) {
		setStale(false);
		return { note, start: a.startLine, end: a.endLine, stale: false, changed };
	}
	if (lines.length <= MAX_SCAN_LINES) {
		const range = a.endLine - a.startLine + 1;
		if (range >= 1 && range <= lines.length) {
			for (let s = 1; s + range - 1 <= lines.length; s++) {
				if (windowHash(lines, s, s + range - 1) === a.hash) {
					a.startLine = s;
					a.endLine = s + range - 1;
					setStale(false);
					return { note, start: s, end: s + range - 1, stale: false, changed: true };
				}
			}
			for (let s = 1; s + range - 1 <= lines.length; s++) {
				if (rangeHash(lines, s, s + range - 1) === a.rangeHash) {
					a.startLine = s;
					a.endLine = s + range - 1;
					setStale(false);
					return { note, start: s, end: s + range - 1, stale: false, changed: true };
				}
			}
		}
	}
	setStale(true);
	return { note, start: a.startLine, end: a.endLine, stale: true, changed };
}

/** Resolve all notes (reading each referenced file once) and persist shifted/stale changes. */
export async function resolveNotes(repoRoot: string, notes: Note[]): Promise<ResolvedNote[]> {
	const fileCache = new Map<string, string[] | null>();
	const resolved: ResolvedNote[] = [];
	for (const note of notes) {
		const abs = resolve(repoRoot, note.path);
		let lines = fileCache.get(abs);
		if (lines === undefined) {
			lines = await readLines(abs);
			fileCache.set(abs, lines);
		}
		resolved.push(resolveNote(note, lines));
	}
	const changed = resolved.filter((r) => r.changed);
	await Promise.all(changed.map((r) => saveNote(repoRoot, r.note)));
	return resolved;
}

function normalizeNote(raw: any): Note {
	return {
		id: String(raw.id),
		path: toPosix(String(raw.path)),
		anchor: raw.anchor,
		text: String(raw.text ?? ""),
		createdAt: String(raw.createdAt ?? ""),
		updatedAt: String(raw.updatedAt ?? ""),
		stale: raw.stale === true,
		archived: raw.archived === true,
	};
}

export async function loadAllNotes(repoRoot: string): Promise<Note[]> {
	const dir = annotationsDir(repoRoot);
	let entries: string[];
	try {
		entries = await readdir(dir);
	} catch {
		return [];
	}
	const notes: Note[] = [];
	await Promise.all(
		entries
			.filter((e) => e.endsWith(".json"))
			.map(async (e) => {
				try {
					const raw = JSON.parse(await readFile(join(dir, e), "utf-8"));
					if (raw && typeof raw === "object" && typeof raw.id === "string" && typeof raw.path === "string" && raw.anchor) {
						notes.push(normalizeNote(raw));
					}
				} catch {
					// skip malformed note files
				}
			}),
	);
	return notes;
}

export async function loadNote(repoRoot: string, id: string): Promise<Note | null> {
	if (!isValidNoteId(id)) return null;
	try {
		const raw = JSON.parse(await readFile(join(annotationsDir(repoRoot), `${id}.json`), "utf-8"));
		return normalizeNote(raw);
	} catch {
		return null;
	}
}

export async function saveNote(repoRoot: string, note: Note): Promise<void> {
	const dir = annotationsDir(repoRoot);
	const file = join(dir, `${note.id}.json`);
	await withFileMutationQueue(file, async () => {
		await mkdir(dir, { recursive: true });
		await writeFile(file, JSON.stringify(note, null, 2) + "\n", "utf-8");
	});
}

export async function deleteNoteFile(repoRoot: string, id: string): Promise<boolean> {
	if (!isValidNoteId(id)) return false;
	const file = join(annotationsDir(repoRoot), `${id}.json`);
	let existed = true;
	await withFileMutationQueue(file, async () => {
		try {
			await rm(file, { force: true });
		} catch {
			existed = false;
		}
	});
	return existed;
}

/** Human/model-readable location of a resolved note. */
export function describeLocation(r: ResolvedNote): string {
	const a = r.note.anchor;
	if (a.type === "file") return "(whole file)";
	if (a.type === "symbol") return `L${r.start} [symbol: ${a.pattern}]`;
	return r.start === r.end ? `L${r.start}` : `L${r.start}–L${r.end}`;
}

export interface AnnotationItem {
	path: string;
	loc: string;
	text: string;
	stale: boolean;
	/** Note id; set by injectors so the relevance gate can cache verdicts per note. */
	id?: string;
}

/** Build the delimited block appended to read/grep results, or null when empty. */
export function formatAnnotationBlock(items: AnnotationItem[]): string | null {
	if (items.length === 0) return null;
	const body = items
		.map((it) => `${it.path} ${it.loc}${it.stale ? " [stale: anchored code no longer matches]" : ""}: ${it.text}`)
		.join("\n");
	let block = `\n─── annotations (${items.length}) ───\n${body}`;
	if (Buffer.byteLength(block, "utf-8") > MAX_ANNOTATION_BLOCK_BYTES) {
		block = block.slice(0, MAX_ANNOTATION_BLOCK_BYTES) + "\n… [annotations truncated — use annotate list to see all]";
	}
	return block;
}
