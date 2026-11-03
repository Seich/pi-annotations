/**
 * Interactive relevance-gate settings editor — bare `/annotations-gate` in TUI mode.
 *
 * Renders <repoRoot>/.pi/annotations/config.json as a SettingsList (the same component
 * pi's own settings menu is built on). Edits are staged in a draft: enum fields cycle on
 * Enter/Space, free-text/number fields open an Input submenu (Enter applies, Esc cancels
 * the field). ctrl+s writes the file and closes the editor; Esc discards the draft.
 * Saving rewrites only the `relevanceGate` key — unknown keys in the JSON survive — and
 * runs the result through parseConfig, so ranges are clamped exactly like a hand edit.
 * The post-save side effects (model setup/warm/kill) live in index.ts, not here.
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
	Input,
	SettingsList,
	matchesKey,
	truncateToWidth,
	type Component,
	type Focusable,
	type SettingItem,
	type SettingsListTheme,
	type TuiMouseEvent,
	type TuiMouseEventResult,
} from "@earendil-works/pi-tui";
import {
	gateConfigToRaw,
	loadConfig,
	parseConfig,
	readRawConfig,
	writeConfig,
	type AnnotationsConfig,
	type RelevanceGateConfig,
} from "./config.ts";

export interface GateEditResult {
	saved: boolean;
	/** Config as parsed (clamped) after the save; undefined when the editor was cancelled. */
	config?: AnnotationsConfig;
}

const BACKEND_VALUES = ["auto", "python", "llama.cpp", "api", "chat", "off"];
/** localModel cycle: "(installed)" = unset → the engine recorded in .gate/setup.json. */
const LOCAL_MODEL_VALUES = ["(installed)", "laya", "julia-1"];
const INSTALLED_LABEL = "(installed)";
const NUMBER_FIELDS = ["maxCandidates", "minItemsToGate", "timeoutMs", "keepThreshold", "ambiguityBand", "taskMaxChars"] as const;

function gateToDraft(gate: RelevanceGateConfig): Record<string, string> {
	return {
		enabled: String(gate.enabled),
		backend: gate.backend,
		localModel: gate.localModel ?? INSTALLED_LABEL,
		model: gate.model ?? "",
		classifierModel: gate.classifierModel ?? "",
		maxCandidates: String(gate.maxCandidates),
		minItemsToGate: String(gate.minItemsToGate),
		timeoutMs: String(gate.timeoutMs),
		keepThreshold: String(gate.keepThreshold),
		ambiguityBand: String(gate.ambiguityBand),
		taskMaxChars: String(gate.taskMaxChars),
	};
}

/** Draft → raw relevanceGate JSON. parseConfig (run on save) clamps/validates every value. */
function draftToRawGate(draft: Record<string, string>): Record<string, unknown> {
	const gate: Record<string, unknown> = { enabled: draft.enabled === "true", backend: draft.backend };
	const model = draft.model.trim();
	if (model) gate.model = model;
	if (draft.localModel && draft.localModel !== INSTALLED_LABEL) gate.localModel = draft.localModel;
	const classifierModel = draft.classifierModel.trim();
	if (classifierModel) gate.classifierModel = classifierModel;
	for (const k of NUMBER_FIELDS) {
		const v = Number(draft[k]);
		if (Number.isFinite(v)) gate[k] = v;
	}
	return gate;
}

/**
 * Open the interactive gate-settings editor. TUI mode only — callers fall back to a
 * text view in other modes. Resolves with { saved: false } when the user discards.
 */
export async function editGateConfig(ctx: ExtensionContext, repoRoot: string): Promise<GateEditResult> {
	const gate = (await loadConfig(repoRoot)).relevanceGate;
	const draft = gateToDraft(gate);
	let dirty = false;
	let finished = false;
	let errorMsg: string | null = null;
	let activeSubmenu: (Component & Focusable) | null = null;
	let focusedValue = false;

	return ctx.ui.custom<GateEditResult>((tui, theme, _kb, done) => {
		const listTheme: SettingsListTheme = {
			label: (text, selected) => (selected ? theme.fg("accent", text) : text),
			value: (text, selected) => (selected ? theme.fg("accent", text) : theme.fg("muted", text)),
			description: (text) => theme.fg("dim", text),
			cursor: theme.fg("accent", "→ "),
			hint: (text) => theme.fg("dim", text),
		};

		const finish = (r: GateEditResult) => {
			if (finished) return;
			finished = true;
			done(r);
		};

		const save = async () => {
			// parseConfig clamps ranges and validates enums exactly like a hand-edited file.
			const parsed = parseConfig({ relevanceGate: draftToRawGate(draft) });
			const raw = await readRawConfig(repoRoot);
			// Merge over the existing object so unknown keys (top level and inside
			// relevanceGate) survive the save.
			const prevGate =
				raw.relevanceGate && typeof raw.relevanceGate === "object" && !Array.isArray(raw.relevanceGate)
					? (raw.relevanceGate as Record<string, unknown>)
					: {};
			raw.relevanceGate = { ...prevGate, ...gateConfigToRaw(parsed.relevanceGate) };
			await writeConfig(repoRoot, raw);
			finish({ saved: true, config: parsed });
		};

		/** Free-text / number field: Enter on the list item opens a single-line Input submenu. */
		const textField = (id: string, description: string, placeholder: string): SettingItem => ({
			id,
			label: id,
			description,
			currentValue: draft[id],
			submenu: (current, onDone) => {
				const input = new Input({ prompt: `${id}: `, placeholder });
				input.setValue(current);
				input.onSubmit = (v) => onDone(v);
				input.onEscape = () => onDone(undefined);
				const comp: Component & Focusable = {
					get focused() {
						return input.focused;
					},
					set focused(v: boolean) {
						input.focused = v;
					},
					render: (w: number) =>
						[
							theme.bold(theme.fg("accent", id)),
							theme.fg("dim", `  ${description}`),
							"",
							...input.render(w),
							"",
							theme.fg("dim", "  enter applies · esc cancels the field · ctrl+s still saves"),
						].map((l) => truncateToWidth(l, w)),
					handleInput: (d: string) => input.handleInput(d),
					invalidate: () => input.invalidate(),
				};
				activeSubmenu = comp;
				return comp;
			},
		});

		const items: SettingItem[] = [
			{
				id: "enabled",
				label: "enabled",
				description: "Master switch — false: annotations inject exactly as without the gate",
				currentValue: draft.enabled,
				values: ["false", "true"],
			},
			{
				id: "backend",
				label: "backend",
				description: "auto tries python → llama.cpp → api → chat (first that works); off = no judging",
				currentValue: draft.backend,
				values: BACKEND_VALUES,
			},
			{
				id: "localModel",
				label: "localModel",
				description: "Engine for the python backend — (installed) uses the engine from .gate/setup.json",
				currentValue: draft.localModel,
				values: LOCAL_MODEL_VALUES,
			},
			textField("model", "chat-backend model override as provider/modelId — empty = current session model", "provider/modelId"),
			textField("classifierModel", "classifier model override for llama.cpp/api backends — empty = auto", "provider/modelId"),
			textField("maxCandidates", "max candidates judged in one gate call — the rest are kept unjudged (fail-open)", "20"),
			textField("minItemsToGate", "below this many candidates the gate is skipped (a lone note is bounded by size)", "2"),
			textField("timeoutMs", "hard wall-clock cap per gate call in ms — on expiry all notes are kept", "5000"),
			textField("keepThreshold", "classifier backends: keep when P(relevant) >= this", "0.5"),
			textField("ambiguityBand", "|P - 0.5| below this counts as unsure → keep (fail-open)", "0.1"),
			textField("taskMaxChars", "cap on the extracted task text sent to the gate (classifier budgets are ~1k tokens)", "2000"),
		];

		let cachedLines: string[] | undefined;
		let cachedWidth = -1;
		// Lines rendered above the list (title, status, blank) — for mouse y-offsets.
		const HEADER_LINES = 3;

		const root: Component & Focusable = {
			get focused() {
				return focusedValue;
			},
			set focused(v: boolean) {
				focusedValue = v;
				if (activeSubmenu) activeSubmenu.focused = v;
			},
			render(width: number): string[] {
				if (cachedLines && cachedWidth === width) return cachedLines;
				cachedWidth = width;
				const status = errorMsg
					? theme.fg("error", `  ${errorMsg}`)
					: dirty
						? theme.fg("warning", "  unsaved changes")
						: theme.fg("dim", "  no changes");
				cachedLines = [
					theme.bold(theme.fg("accent", "Relevance gate settings — .pi/annotations/config.json")),
					`${status} · ${theme.fg("dim", "ctrl+s saves · esc discards")}`,
					"",
					...list.render(width),
				].map((l) => truncateToWidth(l, width));
				return cachedLines;
			},
			handleInput(data: string): void {
				if (matchesKey(data, "ctrl+s")) {
					save().catch((err: any) => {
						errorMsg = `save failed: ${err?.message ?? err}`;
						root.invalidate();
						tui.requestRender();
					});
					return;
				}
				list.handleInput(data);
				root.invalidate();
				tui.requestRender();
			},
			handleMouse(event: TuiMouseEvent): TuiMouseEventResult | undefined {
				const r = list.handleMouse({ ...event, y: event.y - HEADER_LINES });
				if (r?.handled) {
					root.invalidate();
					tui.requestRender();
				}
				return r;
			},
			invalidate(): void {
				cachedLines = undefined;
				list.invalidate();
			},
		};

		const list = new SettingsList(
			items,
			Math.min(items.length, 12),
			listTheme,
			(id, newValue) => {
				draft[id] = newValue;
				dirty = true;
				errorMsg = null;
				root.invalidate();
				tui.requestRender();
			},
			() => finish({ saved: false }),
		);

		return root;
	});
}
