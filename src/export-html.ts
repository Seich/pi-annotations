/**
 * Render source files plus their annotations as standalone, shareable HTML documents
 * with syntax highlighting (highlight.js, when available) and annotation callouts.
 *
 * Two entry points:
 *  - renderAnnotatedHtml(): one file (used by /annotations-export <path>)
 *  - renderRepoHtml(): every annotated file in a single document with a table of
 *    contents (used by /annotations-export with no path); notes for files that no
 *    longer exist render in a "missing files" section so they can be pruned.
 */

import { extname } from "node:path";
import { describeLocation, type ResolvedNote } from "./notes.ts";

type Hljs = {
	highlight: (code: string, opts: { language: string }) => { value: string };
	getLanguage: (lang: string) => unknown;
};

let hljsPromise: Promise<Hljs | null> | null = null;
async function getHljs(): Promise<Hljs | null> {
	hljsPromise ??= import("highlight.js")
		.then((m) => (m.default ?? m) as Hljs)
		.catch(() => null);
	return hljsPromise;
}

const EXT_TO_LANG: Record<string, string> = {
	".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
	".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
	".py": "python", ".rs": "rust", ".go": "go", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
	".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
	".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
	".sh": "bash", ".bash": "bash", ".zsh": "bash",
	".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "ini", ".ini": "ini",
	".md": "markdown", ".html": "xml", ".xml": "xml", ".css": "css", ".scss": "scss",
	".sql": "sql", ".lua": "lua", ".pl": "perl", ".r": "r", ".vue": "xml", ".svelte": "xml",
	".dockerfile": "dockerfile", ".proto": "protobuf", ".graphql": "graphql", ".gql": "graphql",
};

function escapeHtml(s: string): string {
	return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/**
 * Split highlighted HTML into per-line HTML. highlight.js only emits <span> tags, so we
 * close any open spans at each newline and reopen them on the next line — multi-line
 * constructs (block comments, template strings) stay highlighted across lines.
 */
function splitHighlightedLines(html: string): string[] {
	const out: string[] = [];
	let open: string[] = [];
	for (const raw of html.split("\n")) {
		let line = open.join("") + raw;
		for (const tag of raw.match(/<span[^>]*>|<\/span>/g) ?? []) {
			if (tag === "</span>") open.pop();
			else open.push(tag);
		}
		line += "</span>".repeat(open.length);
		out.push(line);
	}
	return out;
}

const CSS = `
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1rem; background: #0d1117; color: #e6edf3;
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.doc { max-width: 1100px; margin: 0 auto; }
header.doc-head { margin-bottom: 1.25rem; }
header.doc-head h1, header.doc-head h2 { font-size: 1.15rem; margin: 0 0 .35rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
header.doc-head .meta { color: #8b949e; font-size: .8rem; }
.legend { display: flex; gap: 1rem; margin-top: .5rem; font-size: .75rem; color: #8b949e; }
.badge { display: inline-block; padding: 0 .4rem; border-radius: 4px; font-size: .7rem;
  font-family: ui-monospace, Menlo, monospace; vertical-align: middle; }
.badge.stale { background: #3d2e00; color: #e3b341; border: 1px solid #9e6a03; }
.badge.archived { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
nav.toc { margin: 0 0 2rem; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: .5rem .6rem; }
nav.toc .toc-title { font-size: .7rem; color: #8b949e; text-transform: uppercase; letter-spacing: .06em; padding: .15rem .35rem .35rem; }
nav.toc a { display: flex; justify-content: space-between; gap: 1rem; padding: .28rem .45rem;
  border-radius: 5px; color: #58a6ff; text-decoration: none; font-size: .8rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
nav.toc a:hover { background: #0d1117; }
nav.toc a .count { color: #8b949e; font-size: .72rem; white-space: nowrap; }
section.file { margin: 2.25rem 0 0; scroll-margin-top: 1rem; }
.back-top { display: inline-block; margin-top: .5rem; color: #6e7681; text-decoration: none; font-size: .72rem; }
.back-top:hover { color: #58a6ff; }
.missing { margin: 2.25rem 0 0; padding: .65rem .8rem .75rem; background: #161b22;
  border: 1px solid #9e6a03; border-radius: 8px; }
.missing h2 { margin: 0 0 .5rem; font-size: .85rem; color: #e3b341; }
.missing .note { margin: .15rem .75rem; }
pre.code { margin: 0; background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
  padding: .75rem 0; overflow-x: auto; font-size: .8rem; line-height: 1.5; }
.line { display: flex; white-space: pre; }
.line:hover { background: #161b22; }
.ln { flex: 0 0 auto; width: 3.5rem; padding-right: 1rem; text-align: right;
  color: #6e7681; user-select: none; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.lc { flex: 1 1 auto; padding-right: 1rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.note { display: flex; gap: .6rem; margin: .15rem .75rem .15rem 3.5rem; padding: .45rem .7rem;
  background: #161b22; border: 1px solid #30363d; border-left: 3px solid #58a6ff;
  border-radius: 6px; font-size: .8rem; }
.note.stale { border-left-color: #e3b341; }
.note.archived { opacity: .55; border-left-color: #6e7681; }
.note .note-loc { flex: 0 0 auto; color: #58a6ff; font-family: ui-monospace, Menlo, monospace;
  font-size: .72rem; padding-top: .1rem; white-space: nowrap; }
.note.stale .note-loc { color: #e3b341; }
.note .note-body { white-space: pre-wrap; word-break: break-word; }
footer.doc-foot { margin-top: 1rem; color: #6e7681; font-size: .72rem; }
/* highlight.js theme (github-dark-ish) */
.hljs { color: #e6edf3; }
.hljs-comment, .hljs-quote { color: #8b949e; font-style: italic; }
.hljs-keyword, .hljs-selector-tag, .hljs-doctag { color: #ff7b72; }
.hljs-string, .hljs-regexp, .hljs-meta .hljs-string { color: #a5d6ff; }
.hljs-number, .hljs-literal { color: #79c0ff; }
.hljs-title, .hljs-title.function_, .hljs-section { color: #d2a8ff; }
.hljs-title.class_, .hljs-type, .hljs-built_in { color: #ffa657; }
.hljs-attr, .hljs-attribute, .hljs-variable, .hljs-template-variable { color: #79c0ff; }
.hljs-name, .hljs-selector-id, .hljs-selector-class { color: #7ee787; }
.hljs-built_in, .hljs-symbol { color: #ffa657; }
.hljs-params { color: #e6edf3; }
.hljs-meta, .hljs-tag { color: #8b949e; }
.hljs-emphasis { font-style: italic; }
.hljs-strong { font-weight: bold; }
`;

const LEGEND_HTML =
	'<div class="legend"><span><span class="badge stale">stale</span> anchored code no longer matches</span>' +
	'<span><span class="badge archived">archived</span> kept on disk, not injected into agent context</span></div>';

function noteCard(rel: string, r: ResolvedNote): string {
	const staleBadge = r.stale ? ` <span class="badge stale">stale: anchored code no longer matches</span>` : "";
	const archivedBadge = r.note.archived ? ` <span class="badge archived">archived</span>` : "";
	return `<div class="note${r.stale ? " stale" : ""}${r.note.archived ? " archived" : ""}">` +
		`<span class="note-loc">${escapeHtml(describeLocation(r))}${staleBadge}${archivedBadge}</span>` +
		`<span class="note-body">${escapeHtml(r.note.text)}</span></div>`;
}

function fileMetaLine(resolved: ResolvedNote[]): string {
	const stale = resolved.filter((r) => r.stale).length;
	const archived = resolved.filter((r) => r.note.archived).length;
	return `${resolved.length} annotation(s)${stale ? `, ${stale} stale` : ""}${archived ? `, ${archived} archived` : ""}`;
}

function fileHeaderHtml(relPath: string, resolved: ResolvedNote[], headingLevel: 1 | 2, includeLegend: boolean): string {
	const fileLevel = resolved.filter((r) => r.note.anchor.type === "file");
	return `<header class="doc-head">
<h${headingLevel}>${escapeHtml(relPath)}</h${headingLevel}>
<div class="meta">${fileMetaLine(resolved)}</div>
${includeLegend ? LEGEND_HTML : ""}
${fileLevel.map((r) => noteCard(relPath, r)).join("\n")}
</header>`;
}

async function codeBlockHtml(relPath: string, code: string, resolved: ResolvedNote[]): Promise<string> {
	const hljs = await getHljs();
	const lang = EXT_TO_LANG[extname(relPath).toLowerCase()];
	let highlighted: string[];
	if (hljs && lang && hljs.getLanguage(lang)) {
		highlighted = splitHighlightedLines(hljs.highlight(code, { language: lang }).value);
	} else {
		highlighted = code.split("\n").map(escapeHtml);
	}

	const lines = code.split("\n");
	const byEnd = new Map<number, ResolvedNote[]>();
	for (const r of resolved) {
		if (r.note.anchor.type === "file") continue;
		const arr = byEnd.get(r.end) ?? [];
		arr.push(r);
		byEnd.set(r.end, arr);
	}

	const body: string[] = [];
	for (let i = 0; i < lines.length; i++) {
		const n = i + 1;
		body.push(`<div class="line"><span class="ln">${n}</span><span class="lc">${highlighted[i] ?? ""}</span></div>`);
		const notes = byEnd.get(n);
		if (notes) for (const r of notes) body.push(noteCard(relPath, r));
	}
	return `<pre class="code">${body.join("")}</pre>`;
}

function shell(title: string, body: string): string {
	return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<style>${CSS}</style>
</head>
<body>
<div class="doc">
${body}
</div>
</body>
</html>
`;
}

export interface ExportHtmlOptions {
	generatedAt?: string;
	extensionName?: string;
}

/** Single-file export: header, legend, file-level notes, highlighted code with inline note cards. */
export async function renderAnnotatedHtml(relPath: string, code: string, resolved: ResolvedNote[], opts: ExportHtmlOptions = {}): Promise<string> {
	const generatedAt = opts.generatedAt ?? new Date().toISOString();
	const header = `<header class="doc-head">
<h1>${escapeHtml(relPath)}</h1>
<div class="meta">${fileMetaLine(resolved)} · generated ${escapeHtml(generatedAt)} by ${escapeHtml(opts.extensionName ?? "pi-annotations")}</div>
${LEGEND_HTML}
${resolved.filter((r) => r.note.anchor.type === "file").map((r) => noteCard(relPath, r)).join("\n")}
</header>`;
	return shell(
		`${relPath} — annotated`,
		`${header}
${await codeBlockHtml(relPath, code, resolved)}
<footer class="doc-foot">Exported with pi-annotations — annotations live in .pi/annotations/ and are injected into agent read/grep results.</footer>`,
	);
}

export interface RepoFile {
	relPath: string;
	code: string;
	resolved: ResolvedNote[];
}

function slug(s: string): string {
	return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "file";
}

/** Whole-repo export: table of contents plus one section per annotated file. */
export async function renderRepoHtml(title: string, files: RepoFile[], missing: ResolvedNote[], opts: ExportHtmlOptions = {}): Promise<string> {
	const generatedAt = opts.generatedAt ?? new Date().toISOString();
	const allResolved = [...files.flatMap((f) => f.resolved), ...missing];
	const totalNotes = allResolved.length;
	const stale = allResolved.filter((r) => r.stale).length;
	const archived = allResolved.filter((r) => r.note.archived).length;

	const toc: string[] = [];
	const sections: string[] = [];
	for (let i = 0; i < files.length; i++) {
		const f = files[i];
		const id = `f${i}-${slug(f.relPath)}`;
		toc.push(
			`<a href="#${id}"><span class="toc-path">${escapeHtml(f.relPath)}</span><span class="count">${fileMetaLine(f.resolved)}</span></a>`,
		);
		sections.push(`<section class="file" id="${id}">
${fileHeaderHtml(f.relPath, f.resolved, 2, false)}
${await codeBlockHtml(f.relPath, f.code, f.resolved)}
<a class="back-top" href="#top">↑ index</a>
</section>`);
	}

	let missingHtml = "";
	if (missing.length > 0) {
		const byPath = new Map<string, ResolvedNote[]>();
		for (const r of missing) {
			const arr = byPath.get(r.note.path) ?? [];
			arr.push(r);
			byPath.set(r.note.path, arr);
		}
		const cards = [...byPath.entries()].map(([p, rs]) => rs.map((r) => noteCard(p, r)).join("\n")).join("\n");
		missingHtml = `<div class="missing"><h2>Notes for missing files (${missing.length})</h2>${cards}</div>`;
	}

	const header = `<header class="doc-head" id="top">
<h1>${escapeHtml(title)} — annotated repo</h1>
<div class="meta">${files.length} file(s) · ${totalNotes} annotation(s)${stale ? `, ${stale} stale` : ""}${archived ? `, ${archived} archived` : ""} · generated ${escapeHtml(generatedAt)} by ${escapeHtml(opts.extensionName ?? "pi-annotations")}</div>
${LEGEND_HTML}
</header>`;

	return shell(
		`${title} — annotated repo`,
		`${header}
<nav class="toc"><div class="toc-title">Files</div>
${toc.join("\n")}
</nav>
${sections.join("\n")}
${missingHtml}
<footer class="doc-foot">Exported with pi-annotations — annotations live in .pi/annotations/ and are injected into agent read/grep results.</footer>`,
	);
}
