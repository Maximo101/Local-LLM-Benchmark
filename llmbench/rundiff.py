"""--diff-runs (v2.4.1): what changed between finished runs, model by model, and whether it is more than noise.

    python benchmark.py --diff-runs                         the two newest finished runs in OUTPUT_DIR
    python benchmark.py --diff-runs 4                       the four newest
    python benchmark.py --diff-runs A B [C ...]             these runs, in this order (the first is the baseline)
    python benchmark.py --diff-runs A B --diff-rescore      score every run again with this version first

A run is a benchmark_<stamp>.json or rescored_<stamp>.json file, a folder holding results.json, or a bare stamp
(20261116_090118 -> OUTPUT_DIR/benchmark_20261116_090118.json).

Unlike --compare, which pools the models of several runs into one ranking, this follows the SAME models across runs:
rank and Agent score, every dataset's accuracy, long context per length, speed, VRAM and power, the launch settings
and the run settings that changed. Nothing is measured, rescored (unless --diff-rescore) or written over; the output
is OUTPUT_DIR/run_diff_<stamp>.html (self-contained, opens offline) and run_diff_<stamp>.json.

Every change between the baseline and the latest run is judged against its noise:
  accuracy   paired when both runs' eval_details_<stamp>.jsonl are found (the same questions: exact McNemar test on
             the questions that flipped), else unpaired (Agresti-Caffo interval for a difference of two proportions);
             95% level either way. Long context per length uses the unpaired test (its codes are drawn per run).
  speed      tokens/s and seconds: a change counts beyond noise at |change| >= max(5%, 2x the larger run-to-run
             spread of the two runs' timed samples).
  scores     Agent score / quality: 0.5 points; VRAM 0.1 GB; power 5%.
Scores compared across tool versions that score differently are flagged; --diff-rescore puts them on one footing.

Only aggregates and counts are read from or written to the report: no question text, answer or reference answer
(safe with GPQA enabled; the report still lives in the git-ignored results/ folder).
"""
from __future__ import annotations

import copy
import glob
import html
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from .common import __version__
from .config import EVAL_SPECS

# ------------------------------------------------------------------------------------------------ what is compared
# (row key, label, unit, better (+1 higher / -1 lower), noise kind)
METRICS = [
    ("score_agent", "Agent score", "", +1, "score"),
    ("agent_quality", "Agent quality", "", +1, "score"),
    ("avg_tps", "Generation", "tok/s", +1, "tps"),
    ("prefill_tps", "Prefill", "tok/s", +1, "tps"),
    ("prefill_64k_tps", "Prefill 64k", "tok/s", +1, "tps"),
    ("prefill_depth_tps", "Prefill at depth", "tok/s", +1, "tps"),
    ("agent_ttft_s", "Agent TTFT", "s", -1, "time"),
    ("eval_avg_sec", "Seconds per question", "s", -1, "time"),
    ("load_s", "Load", "s", -1, "time"),
    ("vram_gb", "VRAM peak", "GB", -1, "vram"),
    ("gpu_w", "GPU power", "W", -1, "power"),
    ("j_per_token", "Energy", "J/tok", -1, "power"),
]
SPEED_KEYS = [m for m in METRICS if m[0] not in ("score_agent", "agent_quality")]
LAUNCH_KEYS = [("ctx_size", "ctx"), ("quant", "quant"), ("kv_type", "KV"), ("n_cpu_moe", "n-cpu-moe"), ("ngl", "ngl"),
               ("ubatch", "ubatch"), ("batch", "batch"), ("parallel", "parallel"), ("flash_attn", "flash-attn"),
               ("offload", "offload"), ("build_info", "llama.cpp build"), ("eval_sampling", "sampling")]
LC_LENGTHS = ("32k", "64k", "90k", "128k")
# config snapshot keys that differ between machines without changing what is measured
_CONFIG_SKIP = {"output_dir", "eval_cache_dir", "gguf_search_dirs", "ui_host", "ui_port", "notify_url"}
SCORE_TOL, VRAM_TOL, POWER_PCT, SPEED_PCT, Z95 = 0.5, 0.1, 5.0, 5.0, 1.959964
BETTER, WORSE, SAME = "better", "worse", "same"


# ------------------------------------------------------------------------------------------------ statistics
def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value: b questions went right -> wrong, c wrong -> right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def agresti_caffo(p1, n1, p2, n2):
    """95% interval for (p2 - p1) in percentage points, Agresti-Caffo (add one right and one wrong to each side).
    p in percent. -> (low, high) or None."""
    if not n1 or not n2 or p1 is None or p2 is None:
        return None
    x1, x2 = p1 / 100 * n1, p2 / 100 * n2
    a1, a2 = (x1 + 1) / (n1 + 2), (x2 + 1) / (n2 + 2)
    se = math.sqrt(a1 * (1 - a1) / (n1 + 2) + a2 * (1 - a2) / (n2 + 2))
    d = a2 - a1
    return 100 * (d - Z95 * se), 100 * (d + Z95 * se)


def kendall_tau(xs, ys):
    """Kendall's tau-b of two equally long score lists (ties allowed). None for fewer than 2 pairs."""
    n = len(xs)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _rel_verdict(a, b, better, pct):
    if a is None or b is None:
        return None
    if a == 0:
        return SAME if b == 0 else (BETTER if (b - a) * better > 0 else WORSE)
    ch = 100 * (b - a) / abs(a)
    if abs(ch) < pct:
        return SAME
    return BETTER if ch * better > 0 else WORSE


def metric_verdict(kind, a, b, better, spread=None):
    """better / worse / same (within noise) / None (one side missing)."""
    a, b = _num(a), _num(b)
    if a is None or b is None:
        return None
    if kind == "score":
        return SAME if abs(b - a) < SCORE_TOL else (BETTER if (b - a) * better > 0 else WORSE)
    if kind == "vram":
        return SAME if abs(b - a) < VRAM_TOL else (BETTER if (b - a) * better > 0 else WORSE)
    if kind == "power":
        return _rel_verdict(a, b, better, POWER_PCT)
    return _rel_verdict(a, b, better, max(SPEED_PCT, 2 * (spread or 0)))


# ------------------------------------------------------------------------------------------------ finding runs
def _out_runs(out_dir):
    """Finished, non-demo benchmark_<stamp>.json files in out_dir, oldest first."""
    found = []
    for f in glob.glob(str(Path(out_dir) / "benchmark_*.json")):
        try:
            doc = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("models") and (doc.get("run") or {}).get("mode") != "demo":
            found.append((str((doc.get("run") or {}).get("id") or ""), os.path.getmtime(f), Path(f)))
    return [p for _, _, p in sorted(found)]


def resolve_runs(cfg, args):
    """Run files to compare, baseline first. See the module docstring for what an argument may be."""
    args = list(args or [])
    if len(args) <= 1 and (not args or (args[0].isdigit() and len(args[0]) <= 2)):
        want = int(args[0]) if args else 2
        if want < 2:
            sys.exit("[error] --diff-runs needs at least two runs.")
        runs = _out_runs(cfg.output_dir)
        if len(runs) < 2:
            sys.exit(f"[error] --diff-runs: {cfg.output_dir} has {len(runs)} finished run(s); two are needed "
                     "(or name the run files).")
        return runs[-want:]
    paths = []
    for a in args:
        p = Path(a).expanduser()
        if p.is_dir():
            p = p / "results.json"
        elif not p.exists() and not p.suffix:                     # a bare stamp
            for name in (f"benchmark_{a}.json", f"rescored_{a}.json"):
                if (cfg.output_dir / name).is_file():
                    p = cfg.output_dir / name
                    break
        if not p.is_file():
            sys.exit(f"[error] --diff-runs: {a} not found (a run JSON, a folder with results.json, or a stamp in "
                     f"{cfg.output_dir}).")
        paths.append(p)
    if len({p.resolve() for p in paths}) < len(paths):
        sys.exit("[error] --diff-runs: the same run was given twice.")
    if len(paths) < 2:
        sys.exit("[error] --diff-runs needs at least two runs.")
    return paths


def load_run(path):
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"[error] --diff-runs: could not read {path}: {e}")
    if not isinstance(doc, dict) or "models" not in doc:
        sys.exit(f"[error] --diff-runs: {path} is not a benchmark results file.")
    return doc


def run_id(doc, path):
    return str((doc.get("run") or {}).get("id") or Path(path).stem.replace("benchmark_", "").replace("rescored_", ""))


def label_of(doc, path):
    rid = run_id(doc, path)
    try:
        when = datetime.strptime(rid, "%Y%m%d_%H%M%S").strftime("%d %b %H:%M")
    except ValueError:
        when = rid
    return when + (" (rescored)" if Path(path).name.startswith("rescored_") else "")


def details_path(doc, path):
    """The run's eval_details_<stamp>.jsonl: as recorded, else next to the JSON (a share mounted elsewhere)."""
    rid = run_id(doc, path)
    rec = (doc.get("artifacts") or {}).get("eval_details")
    cands = ([Path(rec)] if rec else []) + ([Path(path).parent / Path(rec).name] if rec else [])
    cands.append(Path(path).parent / f"eval_details_{rid}.jsonl")
    return next((c for c in cands if c.is_file()), None)


def load_details(p):
    """{(model, dataset, id): correct} from an eval_details file; the last verdict of a question wins, server errors
    are left out (they are not the model's answer). Reads verdicts only, never question or answer text."""
    out = {}
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict) or rec.get("api_error") or rec.get("id") is None:
                    continue
                out[(rec.get("model"), rec.get("dataset"), str(rec.get("id")))] = bool(rec.get("correct"))
    except OSError:
        return None
    return out


# ------------------------------------------------------------------------------------------------ building the diff
def _ranks(rows):
    ranked = sorted((r for r in rows if _num(r.get("score_agent")) is not None), key=lambda r: -r["score_agent"])
    return {r["model"]: i for i, r in enumerate(ranked, 1)}


def _gen_tps(r):
    return _num(r.get("avg_tps")) if _num(r.get("avg_tps")) is not None else _num(r.get("tech_tps"))


def _value(r, key):
    if r is None:
        return None
    return _gen_tps(r) if key == "avg_tps" else _num(r.get(key))


def _lc_n(r, length, default):
    n = _num(r.get(f"longctx_{length}_n"))
    return int(n) if n else default


def _paired(det_a, det_b, model, key):
    if det_a is None or det_b is None:
        return None
    b = c = n = 0
    for (m, d, qid), ok_a in det_a.items():
        if m != model or d != key:
            continue
        ok_b = det_b.get((m, d, qid))
        if ok_b is None:
            continue
        n += 1
        b += ok_a and not ok_b
        c += ok_b and not ok_a
    return {"n": n, "b": b, "c": c, "p": mcnemar_p(b, c)} if n else None


def _acc_entry(vals, ns, paired):
    """Accuracy of one dataset (or length) across runs; delta and verdict are baseline -> latest."""
    a, b, na, nb = vals[0], vals[-1], ns[0], ns[-1]
    e = {"values": vals, "n": ns, "delta": None, "verdict": None, "test": None}
    if a is None or b is None:
        return e
    e["delta"] = b - a
    if paired and paired["n"] >= 10:
        e["test"], e["paired"] = "paired", paired
        e["verdict"] = SAME if paired["p"] >= 0.05 else (BETTER if paired["c"] > paired["b"] else WORSE)
    else:
        ci = agresti_caffo(a, na, b, nb)
        if ci is None:
            return e
        e["test"], e["ci"] = "unpaired", [round(ci[0], 2), round(ci[1], 2)]
        e["verdict"] = SAME if ci[0] <= 0 <= ci[1] else (BETTER if ci[0] > 0 else WORSE)
    return e


def _fmt_launch(v):
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 1024 and float(v).is_integer():
        return f"{int(v)}"
    return "-" if v is None or v == "" else str(v)


def _config_changes(docs):
    snaps = [d.get("config") or {} for d in docs]
    keys = sorted(set().union(*snaps) - _CONFIG_SKIP)
    out = []
    for k in keys:
        vals = [s.get(k) for s in snaps]
        if any(json.dumps(v, sort_keys=True, default=str) != json.dumps(vals[0], sort_keys=True, default=str)
               for v in vals[1:]):
            out.append({"key": k, "values": vals})
    return out


def _scores_version(doc):
    t = doc.get("tool") or {}
    return t.get("rescored_by") or t.get("version") or "?"


def build_diff(docs, paths, rows_list, rescored=False):
    """The whole comparison as one JSON-ready dict. rows_list[i] are the result rows of docs[i] (flat, as
    reload.rows_from_json gives them, rescored or not)."""
    from .compare import check
    labels = [label_of(d, p) for d, p in zip(docs, paths)]
    for i, lbl in enumerate(labels):
        if labels.count(lbl) > 1:
            labels[i] = f"{lbl} [{run_id(docs[i], paths[i])}]"
    problems, notes = check(docs, labels)
    score_versions = [__version__ if rescored else _scores_version(d) for d in docs]
    if len(set(score_versions)) > 1:
        notes.insert(0, "the stored scores come from different versions (" + ", ".join(
            f"{lbl}: v{v}" for lbl, v in zip(labels, score_versions)) + "); add --diff-rescore to score every run with "
            f"v{__version__} so score and rank changes come from the runs, not the scoring code")
    det_paths = [details_path(d, p) for d, p in zip(docs, paths)]
    details = [load_details(p) if p else None for p in det_paths]
    if not all(details):
        missing = [lbl for lbl, d in zip(labels, details) if not d]
        notes.append("eval_details not found for " + ", ".join(missing) + ": accuracy changes use the unpaired test "
                     "(less sensitive than pairing the same questions)")
    by_run = [{r["model"]: r for r in rows} for rows in rows_list]
    ranks = [_ranks(rows) for rows in rows_list]
    last_rank = ranks[-1]
    names = sorted(set().union(*[set(b) for b in by_run]),
                   key=lambda m: (last_rank.get(m, 10 ** 6), min((rk.get(m, 10 ** 6) for rk in ranks)), m))
    ds_keys = [k for k in EVAL_SPECS if any(_num(r.get(f"eval_{k}_pct")) is not None
                                            for rows in rows_list for r in rows)]
    lc_keys = [L for L in LC_LENGTHS if any(_num(r.get(f"longctx_{L}_pct")) is not None
                                             for rows in rows_list for r in rows)]
    lc_default = []
    for d in docs:
        lc = (d.get("evaluation") or {}).get("long_context") or {}
        lc_default.append(int((lc.get("trials") or 2) * (lc.get("keys") or 4)))

    models, acc_up, acc_down, spd_up, spd_down = [], 0, 0, 0, 0
    for m in names:
        rs = [b.get(m) for b in by_run]
        first_i = next(i for i, r in enumerate(rs) if r is not None)
        last_i = max(i for i, r in enumerate(rs) if r is not None)
        rf, rl = rs[first_i], rs[last_i]
        e = {"model": m, "present": [r is not None for r in rs], "status": [r.get("status") if r else None for r in rs],
             "rank": [rk.get(m) for rk in ranks], "state": [r.get("rank_state") if r else None for r in rs],
             "reason": [r.get("rank_reason") if r else None for r in rs], "compared": [first_i, last_i],
             "metrics": {}, "datasets": {}, "longctx": {}, "launch_changes": []}
        both = first_i != last_i
        for key, _label, _unit, better, kind in METRICS:
            vals = [_value(r, key) for r in rs]
            spread = max(_num(rf.get("tps_spread_pct")) or 0, _num(rl.get("tps_spread_pct")) or 0) \
                if kind == "tps" else None
            a, b = vals[first_i], vals[last_i]
            v = metric_verdict(kind, a, b, better, spread) if both else None
            e["metrics"][key] = {"values": vals, "delta": (b - a) if both and a is not None and b is not None else None,
                                 "pct": (100 * (b - a) / abs(a)) if both and a and b is not None else None,
                                 "verdict": v}
            if kind == "tps":
                spd_up += v == BETTER
                spd_down += v == WORSE
        for k in ds_keys:
            vals = [_num(r.get(f"eval_{k}_pct")) if r else None for r in rs]
            ns = [int(_num(r.get(f"eval_{k}_n")) or 0) if r else 0 for r in rs]
            if not any(v is not None for v in vals):
                continue
            if both:
                ent = _acc_entry([vals[first_i], vals[last_i]], [ns[first_i], ns[last_i]],
                                 _paired(details[first_i], details[last_i], m, k))
            else:
                ent = {"delta": None, "verdict": None, "test": None}
            ent["values"], ent["n"] = vals, ns
            e["datasets"][k] = ent
            acc_up += ent["verdict"] == BETTER
            acc_down += ent["verdict"] == WORSE
        avg_vals = [_num(r.get("eval_avg_pct")) if r else None for r in rs]
        if any(v is not None for v in avg_vals) and both:
            tot = [sum(e["datasets"][k]["n"][i] for k in e["datasets"]) for i in (first_i, last_i)]
            ent = _acc_entry([avg_vals[first_i], avg_vals[last_i]], tot, None)
            ent["values"] = avg_vals
            e["eval_avg"] = ent
        for L in lc_keys:
            vals = [_num(r.get(f"longctx_{L}_pct")) if r else None for r in rs]
            if not any(v is not None for v in vals):
                continue
            ns = [_lc_n(r, L, lc_default[i]) if r and vals[i] is not None else 0 for i, r in enumerate(rs)]
            ent = _acc_entry([vals[first_i], vals[last_i]], [ns[first_i], ns[last_i]], None) if both \
                else {"delta": None, "verdict": None, "test": None}
            ent["values"], ent["n"] = vals, ns
            e["longctx"][L] = ent
        if both:
            for key, name in LAUNCH_KEYS:
                a, b = rf.get(key), rl.get(key)
                if a != b and not (a in (None, "") and b in (None, "")):
                    e["launch_changes"].append(f"{name} {_fmt_launch(a)} -> {_fmt_launch(b)}")
        models.append(e)

    agree = []
    for i in range(1, len(docs)):
        common = [m for m in ranks[i - 1] if m in ranks[i]]
        tau = kendall_tau([by_run[i - 1][m]["score_agent"] for m in common],
                          [by_run[i][m]["score_agent"] for m in common])
        agree.append({"from": labels[i - 1], "to": labels[i], "n": len(common),
                      "tau": round(tau, 3) if tau is not None else None})
    present_first, present_last = set(by_run[0]), set(by_run[-1])
    movers = sorted((x for x in models if x["metrics"]["score_agent"]["delta"] is not None),
                    key=lambda x: -abs(x["metrics"]["score_agent"]["delta"]))[:5]
    runs = []
    for d, p, lbl, det, rows, sv in zip(docs, paths, labels, det_paths, rows_list, score_versions):
        run, ev = d.get("run") or {}, d.get("evaluation") or {}
        builds = sorted({str(r.get("build_info")) for r in rows if r.get("build_info")})
        runs.append({"label": lbl, "id": run_id(d, p), "file": Path(p).name, "version": (d.get("tool") or {}).get("version"),
                     "scored_by": sv, "backend": run.get("backend"), "status": run.get("status"),
                     "started_at": run.get("started_at"), "duration_s": run.get("duration_s"),
                     "hardware": (d.get("host") or {}).get("hardware"), "profile": ev.get("profile"),
                     "models": len(rows), "ranked": sum(1 for r in rows if _num(r.get("score_agent")) is not None),
                     "builds": builds, "details": bool(det)})
    return {"schema": 1, "tool": {"name": "Local-LLM-Benchmark", "version": __version__},
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "scores": f"rescored with v{__version__}" if rescored else "as stored by each run",
            "runs": runs, "problems": problems, "notes": notes, "config_changes": _config_changes(docs),
            "rank_agreement": agree, "datasets": ds_keys, "longctx_lengths": lc_keys,
            "summary": {"models": len(names), "common": len(present_first & present_last),
                        "new": sorted(present_last - present_first), "dropped": sorted(present_first - present_last),
                        "movers": [{"model": x["model"], "delta": round(x["metrics"]["score_agent"]["delta"], 2),
                                    "rank": [x["rank"][x["compared"][0]], x["rank"][x["compared"][1]]]}
                                   for x in movers],
                        "accuracy_better": acc_up, "accuracy_worse": acc_down,
                        "speed_better": spd_up, "speed_worse": spd_down},
            "models": models}


# ------------------------------------------------------------------------------------------------ HTML
_E = html.escape
_GLYPH = {BETTER: "↑", WORSE: "↓", SAME: "≈"}

_CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--ring:rgba(11,11,11,.10);--up:#006300;--down:#d03b3b;--warn:#8a5a00;--line:#2a78d6;--hl:#eef4fc}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--up:#0ca30c;--down:#e06060;--warn:#fab219;
--line:#3987e5;--hl:#1d2a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px 20px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:32px 0 8px}.sub{color:var(--ink2);margin:0 0 16px}.muted{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:12px 16px;margin:10px 0}
.banner{border-left:4px solid var(--down)}.banner.note{border-left-color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tile .v{font-size:24px;font-weight:600}.tile .k{color:var(--ink2);font-size:12px}
.wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--ring);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 9px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--surface);font-weight:600;color:var(--ink2);cursor:pointer;user-select:none}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--surface)}
tr:hover td{background:var(--hl)}td.l{text-align:left;white-space:normal;max-width:420px}
.better{color:var(--up);font-weight:600}.worse{color:var(--down);font-weight:600}.same{color:var(--muted)}
.na{color:var(--muted)}.small{font-size:12px}.tag{display:inline-block;padding:0 6px;border-radius:9px;
border:1px solid var(--ring);font-size:11px;color:var(--ink2);margin-left:4px}
.ctl{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:8px 0}
input[type=search]{padding:5px 8px;border:1px solid var(--ring);border-radius:6px;background:var(--surface);
color:var(--ink);min-width:240px}svg.spark{vertical-align:middle}svg.spark polyline{fill:none;stroke:var(--line);
stroke-width:2;stroke-linejoin:round}svg.spark circle{fill:var(--line);stroke:var(--surface);stroke-width:2}
code{font-size:12px}ul{margin:4px 0;padding-left:20px}
"""

_JS = r"""
(function(){
  function key(td){var v=td.getAttribute('data-v');if(v===null||v==='')return null;var n=parseFloat(v);
    return isNaN(n)?v.toLowerCase():n;}
  document.querySelectorAll('table.sortable').forEach(function(t){
    t.querySelectorAll('th').forEach(function(th,i){th.addEventListener('click',function(){
      var asc=th.getAttribute('data-dir')!=='asc';t.querySelectorAll('th').forEach(function(h){h.removeAttribute('data-dir');});
      th.setAttribute('data-dir',asc?'asc':'desc');var tb=t.tBodies[0];var rows=Array.prototype.slice.call(tb.rows);
      rows.sort(function(a,b){var x=key(a.cells[i]),y=key(b.cells[i]);if(x===null&&y===null)return 0;
        if(x===null)return 1;if(y===null)return -1;if(x<y)return asc?-1:1;if(x>y)return asc?1:-1;return 0;});
      rows.forEach(function(r){tb.appendChild(r);});});});});
  var q=document.getElementById('q'),only=document.getElementById('only');
  function filt(){var s=(q.value||'').toLowerCase(),o=only.checked;
    document.querySelectorAll('tr[data-model]').forEach(function(r){
      var hit=r.getAttribute('data-model').toLowerCase().indexOf(s)>=0;
      var ch=r.getAttribute('data-change')==='1';r.style.display=(hit&&(!o||ch))?'':'none';});}
  if(q){q.addEventListener('input',filt);only.addEventListener('change',filt);}
})();
"""


def _fmt(v, unit=""):
    if v is None:
        return "-"
    if unit in ("tok/s", "W") or abs(v) >= 100:
        return f"{v:,.0f}"
    if unit == "GB" or abs(v) < 10:
        return f"{v:.2f}"
    return f"{v:.1f}"


def _delta_cell(ent, unit="", pct=False, title=""):
    """A <td> with the signed change and a better / worse / within-noise glyph; data-v sorts by the change."""
    v, verdict = ent.get("pct" if pct else "delta"), ent.get("verdict")
    if v is None:
        return f'<td class="na" data-v="" title="{_E(title)}">-</td>'
    cls = verdict or "na"
    txt = (f"{v:+.1f}%" if pct else f"{v:+.1f}" if unit in ("", "pts") else f"{v:+.2f}")
    glyph = _GLYPH.get(verdict, "")
    return f'<td class="{cls}" data-v="{v:.4f}" title="{_E(title)}">{_E(txt)} {glyph}</td>'


def _spark(vals, w=74, h=20):
    pts = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(pts) < 2:
        return ""
    lo, hi = min(v for _, v in pts), max(v for _, v in pts)
    span, n = (hi - lo) or 1.0, max(len(vals) - 1, 1)
    xy = [(4 + i * (w - 8) / n, h - 4 - (v - lo) / span * (h - 8)) for i, v in pts]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3"><title>{v:.1f}</title></circle>'
                   for (x, y), (_, v) in zip(xy, pts))
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Agent score per run">'f'<polyline points="{poly}"/>{dots}</svg>')


def _changed(e):
    parts = list(e["metrics"].values()) + list(e["datasets"].values()) + list(e["longctx"].values())
    own = [c for c in e["launch_changes"] if not c.startswith("llama.cpp build")]   # a build change hits every model
    return any(p.get("verdict") in (BETTER, WORSE) for p in parts) or bool(own) \
        or len(set(x for x in e["state"] if x is not None)) > 1 or not all(e["present"])


def _row_open(e):
    return f'<tr data-model="{_E(e["model"])}" data-change="{1 if _changed(e) else 0}">' \
           f'<td data-v="{_E(e["model"].lower())}">{_E(e["model"])}</td>'


def render_html(diff):
    runs, labels = diff["runs"], [r["label"] for r in diff["runs"]]
    first, last = labels[0], labels[-1]
    out = ['<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; "
           "script-src 'unsafe-inline'; img-src data:\">",
           f"<title>Run diff: {_E(' → '.join(labels))}</title><style>{_CSS}</style></head><body><main>",
           f"<h1>Run diff: {_E(' → '.join(labels))}</h1>",
           f'<p class="sub">Local-LLM-Benchmark v{_E(diff["tool"]["version"])} · generated {_E(diff["generated_at"])} · '
           f'scores {_E(diff["scores"])} · changes are <b>{_E(first)} → {_E(last)}</b> '
           "(a model missing from one of them is compared over the runs it is in).</p>"]
    if diff["problems"]:
        out.append('<div class="card banner"><b>Not like-for-like:</b> these runs did not ask or grade the questions '
                   "the same way, so accuracy and score changes partly come from the settings.<ul>"
                   + "".join(f"<li>{_E(x)}</li>" for x in diff["problems"]) + "</ul></div>")
    if diff["notes"]:
        out.append('<div class="card banner note"><b>Notes</b><ul>'
                   + "".join(f"<li>{_E(x)}</li>" for x in diff["notes"]) + "</ul></div>")

    s = diff["summary"]
    tau = [a for a in diff["rank_agreement"] if a["tau"] is not None]
    tiles = [(str(s["models"]), "models in any run"), (str(s["common"]), f"in both {first} and {last}"),
             (f'{len(s["new"])} / {len(s["dropped"])}', "new / dropped models"),
             (f'{tau[-1]["tau"]:+.2f}' if tau else "-",
              f"rank agreement (Kendall τ, {tau[-1]['n']} models)" if tau else "rank agreement"),
             (f'{s["accuracy_better"]} ↑ · {s["accuracy_worse"]} ↓', "dataset results beyond noise"),
             (f'{s["speed_better"]} ↑ · {s["speed_worse"]} ↓', "tokens/s results beyond noise")]
    out.append('<div class="tiles">' + "".join(f'<div class="card tile"><div class="v">{_E(v)}</div>'
                                               f'<div class="k">{_E(k)}</div></div>' for v, k in tiles) + "</div>")
    out.append('<p class="small muted">↑ better beyond noise · ↓ worse beyond noise · ≈ within noise · '
               "- not measured in one of the runs. Hover a cell for the values in every run and the test used.</p>")

    out.append("<h2>Runs</h2><div class=\"wrap\"><table><thead><tr><th>Run</th><th>File</th><th>Version</th>"
               "<th>Scored by</th><th>Status</th><th>Backend</th><th>Profile</th><th>Models</th><th>Ranked</th>"
               "<th>Duration</th><th>llama.cpp builds</th><th>Per-question details</th><th>Hardware</th></tr></thead><tbody>")
    for r in runs:
        dur = f'{r["duration_s"] / 3600:.1f} h' if _num(r.get("duration_s")) else "-"
        out.append(f'<tr><td>{_E(r["label"])}</td><td class="l"><code>{_E(r["file"])}</code></td>'
                   f'<td>{_E(str(r["version"] or "?"))}</td><td>{_E(str(r["scored_by"]))}</td>'
                   f'<td>{_E(str(r["status"] or "?"))}</td><td>{_E(str(r["backend"] or "?"))}</td>'
                   f'<td>{_E(str(r["profile"] or "-"))}</td><td>{r["models"]}</td><td>{r["ranked"]}</td>'
                   f'<td>{dur}</td><td class="l">{_E(", ".join(r["builds"]) or "-")}</td>'
                   f'<td>{"yes" if r["details"] else "no"}</td><td class="l">{_E(str(r["hardware"] or "-"))}</td></tr>')
    out.append("</tbody></table></div>")

    out.append('<div class="ctl"><input id="q" type="search" placeholder="Filter models" aria-label="Filter models">'
               '<label><input id="only" type="checkbox"> only models with a change beyond noise, a new state or new '
               "launch settings</label></div>")

    # ---- ranking
    spark = len(runs) >= 3
    out.append("<h2>Agent ranking</h2><div class=\"wrap\"><table class=\"sortable\"><thead><tr><th>Model</th>")
    out.extend(f"<th>{_E(l)}<br><span class=\"small muted\">rank · score</span></th>" for l in labels)
    out.append(("<th>Trend</th>" if spark else "") + "<th>Δ score</th><th>Δ rank</th><th>Δ quality</th>"
               "<th>State</th><th>Launch changes</th></tr></thead><tbody>")
    for e in diff["models"]:
        out.append(_row_open(e))
        sc = e["metrics"]["score_agent"]["values"]
        for i, l in enumerate(labels):
            if not e["present"][i]:
                out.append('<td class="na" data-v="">absent</td>')
                continue
            rk, st = e["rank"][i], e["state"][i]
            txt = f"#{rk} · {sc[i]:.1f}" if rk else ("screened" if st == "screened" else
                                                      "no Agent score" if st == "no_agent" else
                                                      e["status"][i] if e["status"][i] != "ok" else "-")
            out.append(f'<td data-v="{rk if rk else ""}" title="{_E(e["reason"][i] or "")}">{_E(txt)}</td>')
        if spark:
            out.append(f'<td data-v="">{_spark(sc)}</td>')
        out.append(_delta_cell(e["metrics"]["score_agent"], title="Agent score " + " → ".join(_fmt(v) for v in sc)))
        a, b = (e["rank"][i] for i in e["compared"])
        if a and b and a != b:
            cls = BETTER if b < a else WORSE
            out.append(f'<td class="{cls}" data-v="{a - b}">{a - b:+d} {_GLYPH[cls]}</td>')
        else:
            out.append(f'<td class="{"same" if a and b else "na"}" data-v="{0 if a and b else ""}">'
                       f'{"0" if a and b else "-"}</td>')
        out.append(_delta_cell(e["metrics"]["agent_quality"]))
        states = [x or "-" for x, p in zip(e["state"], e["present"]) if p]
        st = " → ".join(dict.fromkeys(states)) if len(set(states)) > 1 else states[0] if states else "-"
        out.append(f'<td class="l small" data-v="{_E(st)}">{_E(st)}</td>')
        out.append(f'<td class="l small" data-v="{len(e["launch_changes"])}">{_E("; ".join(e["launch_changes"]) or "-")}</td></tr>')
    out.append("</tbody></table></div>")
    out.append('<p class="small muted">The Agent score and quality are relative to the best finalist of their own run '
               "(the tie zone), so they can move when other models come, go or change; the accuracy, long-context and "
               "speed tables below are each model's own results.</p>")

    # ---- accuracy
    dk = diff["datasets"]
    if dk:
        out.append(f"<h2>Accuracy (percentage points, {_E(first)} → {_E(last)})</h2>"
                   '<div class="wrap"><table class="sortable"><thead><tr><th>Model</th><th>Average</th>')
        out.extend(f"<th>{_E(EVAL_SPECS.get(k, {}).get('short', k))}</th>" for k in dk)
        out.append("</tr></thead><tbody>")
        for e in diff["models"]:
            out.append(_row_open(e))
            av = e.get("eval_avg") or {}
            out.append(_delta_cell(av, title=_acc_title("average", av, labels)))
            for k in dk:
                ent = e["datasets"].get(k) or {}
                out.append(_delta_cell(ent, title=_acc_title(EVAL_SPECS.get(k, {}).get("short", k), ent, labels)))
            out.append("</tr>")
        out.append("</tbody></table></div>")

    # ---- long context
    lk = diff["longctx_lengths"]
    if lk:
        out.append("<h2>Long-context retrieval (percentage points)</h2><div class=\"wrap\"><table class=\"sortable\">"
                   "<thead><tr><th>Model</th>" + "".join(f"<th>{_E(L)}</th>" for L in lk) + "</tr></thead><tbody>")
        for e in diff["models"]:
            if not e["longctx"]:
                continue
            out.append(_row_open(e))
            for L in lk:
                ent = e["longctx"].get(L) or {}
                out.append(_delta_cell(ent, title=_acc_title(L, ent, labels)))
            out.append("</tr>")
        out.append("</tbody></table></div>")

    # ---- speed and cost
    out.append(f"<h2>Speed and cost (change, {_E(first)} → {_E(last)})</h2>"
               '<div class="wrap"><table class="sortable"><thead><tr><th>Model</th>')
    out.extend(f"<th>{_E(lbl)}<br><span class=\"small muted\">{_E(u)}</span></th>" for _, lbl, u, _, _ in SPEED_KEYS)
    out.append("</tr></thead><tbody>")
    for e in diff["models"]:
        out.append(_row_open(e))
        for key, lbl, unit, _, kind in SPEED_KEYS:
            ent = e["metrics"][key]
            vals = " → ".join(_fmt(v, unit) for v in ent["values"])
            if kind == "vram":
                out.append(_delta_cell(ent, unit="GB", title=f"{lbl} ({unit}): {vals}"))
            else:
                out.append(_delta_cell(ent, pct=True, title=f"{lbl} ({unit}): {vals}"))
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # ---- settings
    if diff["config_changes"]:
        out.append("<h2>Settings that differ</h2><div class=\"wrap\"><table><thead><tr><th>Setting</th>"
                   + "".join(f"<th>{_E(l)}</th>" for l in labels) + "</tr></thead><tbody>")
        for c in diff["config_changes"]:
            out.append(f'<tr><td><code>{_E(c["key"].upper())}</code></td>' + "".join(
                f'<td class="l small">{_E(_short(v))}</td>' for v in c["values"]) + "</tr>")
        out.append("</tbody></table></div>")
    if diff["rank_agreement"]:
        out.append('<p class="small muted">Rank agreement between consecutive runs (Kendall τ-b over the models ranked '
                   "in both; 1 = same order, 0 = unrelated): " + "; ".join(
                       f'{_E(a["from"])} → {_E(a["to"])}: '
                       + (f'{a["tau"]:+.2f} ({a["n"]} models)' if a["tau"] is not None else "too few models")
                       for a in diff["rank_agreement"]) + ".</p>")
    out.append('<h2>How changes are judged</h2><div class="card small"><ul>'
               "<li><b>Accuracy:</b> paired exact McNemar test (McNemar 1947) on the questions both runs asked when "
               "their <code>eval_details</code> files are found, else the Agresti-Caffo interval for a difference of two "
               "proportions (Agresti &amp; Caffo 2000); 95% level. Long context uses the unpaired test.</li>"
               f"<li><b>Speed and time:</b> beyond noise at |change| ≥ max({SPEED_PCT:.0f}%, 2 × the larger run-to-run "
               f"spread of the timed samples). Power and energy: {POWER_PCT:.0f}%. VRAM: {VRAM_TOL} GB. Agent score and "
               f"quality: {SCORE_TOL} points.</li>"
               "<li><b>Rank agreement:</b> Kendall's τ-b (Kendall 1938, 1945) over the Agent scores of models ranked in "
               "both runs.</li><li>A llama.cpp build change, new launch settings or new run settings can explain a change; "
               "they are listed above. Nothing here changes a score.</li></ul></div>")
    out.append(f"<script>{_JS}</script></main></body></html>")
    return "\n".join(out)


def _short(v, n=90):
    s = "-" if v is None else json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
    return s if len(s) <= n else s[:n - 1] + "…"


def _acc_title(name, ent, labels):
    if not ent or not ent.get("values"):
        return f"{name}: not measured"
    vals = " → ".join("-" if v is None else f"{v:.1f}% (n={n})" for v, n in zip(ent["values"], ent["n"]))
    t = f"{name}: {vals}"
    if ent.get("test") == "paired":
        p = ent["paired"]
        t += (f". Paired over {p['n']} shared questions: {p['b']} right→wrong, {p['c']} wrong→right, "
              f"McNemar p = {p['p']:.3f}")
    elif ent.get("test") == "unpaired":
        t += f". 95% interval of the change: {ent['ci'][0]:+.1f} to {ent['ci'][1]:+.1f} points"
    return t


# ------------------------------------------------------------------------------------------------ entry point
def _print_summary(diff, html_path, json_path):
    s, labels = diff["summary"], [r["label"] for r in diff["runs"]]
    print(f"Run diff: {' -> '.join(labels)} ({diff['scores']})")
    for x in diff["problems"]:
        print(f"  [not like-for-like] {x}")
    for x in diff["notes"]:
        print(f"  [note] {x}")
    print(f"  {s['models']} models, {s['common']} in both {labels[0]} and {labels[-1]}"
          + (f"; new: {', '.join(s['new'])}" if s["new"] else "")
          + (f"; dropped: {', '.join(s['dropped'])}" if s["dropped"] else ""))
    for a in diff["rank_agreement"]:
        if a["tau"] is not None:
            print(f"  rank agreement {a['from']} -> {a['to']}: Kendall tau {a['tau']:+.2f} over {a['n']} models")
    print(f"  dataset results beyond noise: {s['accuracy_better']} better, {s['accuracy_worse']} worse; "
          f"tokens/s beyond noise: {s['speed_better']} better, {s['speed_worse']} worse")
    if s["movers"]:
        print("  Biggest Agent score changes:")
        for m in s["movers"]:
            rk = " -> ".join("-" if r is None else f"#{r}" for r in m["rank"])
            print(f"    {m['delta']:+6.2f}  {rk:<12} {m['model']}")
    print(f"Report: {html_path}\nJSON:   {json_path}")


def diff_runs(cfg, args=None, rescore=False):
    """--diff-runs: -> (html path, json path, diff dict)."""
    from .reload import config_from_json, rows_from_json
    from .scoring import finalize
    paths = resolve_runs(cfg, args)
    docs = [load_run(p) for p in paths]
    rows_list = []
    for doc in docs:
        rows = rows_from_json(doc)
        if rescore:
            c = config_from_json(copy.deepcopy(cfg), doc)
            finalize(rows, c)
        rows_list.append(rows)
    diff = build_diff(docs, paths, rows_list, rescored=rescore)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    hp, jp = cfg.output_dir / f"run_diff_{stamp}.html", cfg.output_dir / f"run_diff_{stamp}.json"
    from .reporting.json_out import _atomic_write, _clean
    _atomic_write(hp, render_html(diff))
    _atomic_write(jp, json.dumps(_clean(diff), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _print_summary(diff, hp, jp)
    return hp, jp, diff
