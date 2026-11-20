"""Synthetic finished runs for tests of tools that read runs (--diff-runs, --compare, --rescore): a run JSON written
by the real json_out code, plus a matching eval_details_<stamp>.jsonl with one verdict per question. No server, GPU or
dataset is involved; every number is chosen by the test."""
from __future__ import annotations

import copy
import json

from llmbench.diagnostics import demo_results
from llmbench.reporting.json_out import write_json
from llmbench.scoring import finalize

# never shown by a report: written into every detail line so a test can check that no question text leaks
SECRET = "GPQA-QUESTION-TEXT-MUST-NOT-LEAK"


def base_rows(cfg):
    """The six --demo models (deterministic), as result rows."""
    return demo_results(cfg)


def verdicts(rows, flips=None):
    """{(model, dataset, qid): correct} consistent with each row's eval_<k>_pct / _n: the first round(pct x n) ids are
    right. flips {(model, dataset): k} turns the first k wrong ids right (k > 0) or the last |k| right ids wrong."""
    out = {}
    for r in rows:
        for k in [k[5:-4] for k in r if k.startswith("eval_") and k.endswith("_pct") and k != "eval_avg_pct"]:
            n, pct = r.get(f"eval_{k}_n"), r.get(f"eval_{k}_pct")
            if not n or pct is None:
                continue
            right = round(pct * n / 100)
            ok = [i < right for i in range(n)]
            f = (flips or {}).get((r["model"], k), 0)
            for i in (range(right, min(n, right + f)) if f > 0 else range(max(0, right + f), right)):
                ok[i] = f > 0
            for i in range(n):
                out[(r["model"], k, f"{k}-{i}")] = ok[i]
            r[f"eval_{k}_pct"] = 100.0 * sum(ok) / n
    return out


def write_run(cfg, rows, stamp, details=None, version=None, **cfg_over):
    """Score `rows` with this version, write benchmark_<stamp>.json (+ eval_details_<stamp>.jsonl when `details`).
    Returns the JSON path. `version` overrides the tool version recorded in the JSON."""
    run_cfg = copy.deepcopy(cfg)
    for k, v in cfg_over.items():
        setattr(run_cfg, k, v)
    rows = copy.deepcopy(rows)
    finalize(rows, run_cfg)
    arts = {}
    if details is not None:
        dp = cfg.output_dir / f"eval_details_{stamp}.jsonl"
        with open(dp, "w", encoding="utf-8") as fh:
            for (m, d, qid), ok in details.items():
                fh.write(json.dumps({"model": m, "dataset": d, "id": qid, "correct": ok, "na": False,
                                     "answer": SECRET, "gold": SECRET, "response": SECRET, "api_error": False,
                                     "stage": "final"}) + "\n")
        arts["eval_details"] = str(dp)
    write_json(rows, run_cfg, {"stamp": stamp, "status": "complete", "hardware": "Test host", "cpu_source": "demo",
                               "mode": "full", "artifacts": arts}, stamp)
    path = cfg.output_dir / f"benchmark_{stamp}.json"
    if version:
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["tool"]["version"] = version
        path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def two_runs(cfg, details=True):
    """A baseline and a later run of the demo models with known changes:
      model 0: 20 GSM8K questions wrong -> right (a real gain), llama.cpp build changed, generation 20% slower;
      model 1: 2 MMLU-Pro questions right -> wrong (noise);
      model 2: context 65536 -> 131072 (a launch change);
      model 5: dropped; a new model "New-14B-Q4_K_M.gguf" appears.
    -> (path A, path B, rows A, rows B)"""
    a = base_rows(cfg)
    for r in a:
        r["build_info"] = "b12316"
        r["eval_gsm8k_pct"] = min(r["eval_gsm8k_pct"], 50.0)           # room for 20 more right answers out of 50
        r["tps_spread_pct"] = 1.0
    da = verdicts(a)
    b = copy.deepcopy(a[:5])
    for r in b:
        r["build_info"] = "b12407"
    m0, m1 = b[0]["model"], b[1]["model"]
    for key in ("tech_tps", "logic_tps", "avg_tps"):
        if b[0].get(key):
            b[0][key] *= 0.8
    b[2]["ctx_size"] = 131072
    new = copy.deepcopy(a[3])
    new["model"], new["build_info"] = "New-14B-Q4_K_M.gguf", "b12407"
    b.append(new)
    db = verdicts(b, {(m0, "gsm8k"): 20, (m1, "mmlu_pro"): -2})
    pa = write_run(cfg, a, "20261116_090118", da if details else None)
    pb = write_run(cfg, b, "20261119_115908", db if details else None)
    return pa, pb, a, b
