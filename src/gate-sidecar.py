#!/usr/bin/env python3
"""Stdio JSON-RPC sidecar for the pi-annotations relevance gate.

Runs a local "decision model" (Julia-1 or Laya) and answers typed questions
about JSON state, in pi's ClassifierContext shape — no HTTP, no auth, no
network beyond the one-time weight download.

Protocol (one JSON per line on stdin/stdout):
  request:  {"id": 1, "ping": true}
  request:  {"id": 2, "state": {...}, "questions": {...}}   # pi ClassifierContext
  response: {"ready": true, "engine": "...", "model": "..."}  # once, after load
  response: {"id": 1, "pong": true}
  response: {"id": 2, "answers": {...}}  or  {"id": 2, "error": "..."}

Answers are normalized to pi's ClassifierAnswer shapes:
  choice -> {"type": "choice", "choice", "probabilities", "confidence"}
  score  -> {"type": "score", "score", "confidence"}
  bool   -> {"type": "bool", "probability"}   (probability that the answer is true)

pi's "bool" question maps to the engines' native "noul" (yes/no probability).
"""

import argparse
import json
import sys
import traceback
import warnings

# laya's typed-decisions checkpoint ships out-of-range calibration temperatures and warns on
# load (it clamps them itself); huggingface_hub prints download progress bars to stderr.
# Neither is an error — keep our stderr clean for the parent's error forwarding.
warnings.filterwarnings("ignore")
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def to_engine_question(q):
    """pi ClassifierQuestion -> engine question (laya/julia native shape)."""
    t = q.get("type")
    out = {"type": t, "instructions": q.get("instructions", "")}
    crit = q.get("criteria")
    if t == "bool":
        out["type"] = "noul"
        if isinstance(crit, dict):
            out["criteria"] = {"true": crit.get("true", "yes"), "false": crit.get("false", "no")}
    elif crit is not None:
        out["criteria"] = crit
    return out


def normalize_answer(qtype, raw):
    """Engine answer -> pi ClassifierAnswer. Raises on unreadable shapes."""
    if raw is None:
        raise ValueError("engine returned no answer")
    if qtype == "bool":
        p = None
        if isinstance(raw, dict):
            for k in ("noul", "probability", "yes", "true", "value"):
                if k in raw:
                    v = raw[k]
                    p = v.get("probability") if isinstance(v, dict) else v
                    break
        elif isinstance(raw, (int, float)):
            p = raw
        if not isinstance(p, (int, float)):
            raise ValueError(f"cannot read noul/bool answer: {raw!r}")
        return {"type": "bool", "probability": float(p)}
    if qtype == "choice":
        if not isinstance(raw, dict):
            raise ValueError(f"cannot read choice answer: {raw!r}")
        choice = raw.get("choice")
        probs = raw.get("probabilities") or {}
        if not probs and choice is not None:
            probs = {str(choice): 1.0}
        conf = raw.get("confidence")
        if conf is None:
            conf = max(probs.values()) if probs else 0.0
        return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": float(conf)}
    if qtype == "score":
        if not isinstance(raw, dict):
            raise ValueError(f"cannot read score answer: {raw!r}")
        return {"type": "score", "score": float(raw.get("score", 0)), "confidence": float(raw.get("confidence", 0))}
    raise ValueError(f"unknown question type: {qtype!r}")


class LayaEngine:
    """Laya via the official `laya` package (weights auto-download from HF, pin with LAYA_REVISION).

    Uses the typed-decisions checkpoint explicitly: the base English checkpoint scores near
    chance (~36%) on typed questions, typed-decisions is the one trained for choice/score/noul.
    """

    def __init__(self, model):
        import laya

        self.agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions")

    def predict(self, state, questions):
        eng_q = {k: to_engine_question(v) for k, v in questions.items()}
        res = self.agent.predict(json.dumps(state, ensure_ascii=False), eng_q)
        return res["answers"]


class JuliaEngine:
    """Julia-1 via SupersonicLabs' runtime; weights pre-downloaded to --model-dir."""

    def __init__(self, model_dir):
        from julia import load_model

        self.engine = load_model(model_dir, device="cpu", strict_encoding=False, max_length=8192, head_length=512)

    def predict(self, state, questions):
        eng_q = {k: to_engine_question(v) for k, v in questions.items()}
        res = self.engine.predict(state=json.dumps(state, ensure_ascii=False), questions=eng_q)
        return res["answers"]


ENGINES = {"laya": LayaEngine, "julia-1": JuliaEngine}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--model-dir", default=None, help="local weights dir (julia-1)")
    args = ap.parse_args()

    if args.engine == "julia-1" and not args.model_dir:
        sys.stderr.write("julia-1 requires --model-dir\n")
        sys.exit(2)

    engine = ENGINES[args.engine](args.model_dir)
    sys.stdout.write(json.dumps({"ready": True, "engine": args.engine}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = None
        try:
            req = json.loads(line)
            if req.get("ping"):
                resp = {"id": req.get("id"), "pong": True}
            else:
                questions = req.get("questions") or {}
                raw = engine.predict(req.get("state") or {}, questions)
                answers = {name: normalize_answer(q.get("type"), raw.get(name)) for name, q in questions.items()}
                resp = {"id": req.get("id"), "answers": answers}
        except Exception as e:  # noqa: BLE001 - report any failure to the parent
            resp = {"id": req.get("id") if req else None, "error": f"{e}\n{traceback.format_exc()}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
