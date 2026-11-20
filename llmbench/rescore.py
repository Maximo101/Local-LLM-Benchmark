"""--rescore (v2.4): score a finished run again with the current scoring code, and show what changed.

`--reload` keeps a run's stored scores (it is for working on the dashboard). `--rescore` recomputes them: the run's
measurements and answers are kept as they are, and only the scoring (Agent score, quality, tie zone, usable context,
rank states, the other scores) is done again by this version. Nothing is benchmarked and no server is needed.

Settings: the run's own (from its JSON). A scoring setting the run did not record (one added after it was made, such
as AGENT_LONGCTX_BASIS in v2.4) takes its value from today's .env or its default; the table header says which.

Output: an old-vs-new table in the terminal, and in OUTPUT_DIR
  rescored_<run id>.json                       the run's JSON with the new scores (not named benchmark_*.json, so
                                               --estimate, --reload and --compare never mistake it for a run)
  benchmark_dashboard_<run id>_rescored.html   its dashboard
The run's own files, results.json and the latest dashboard are left alone.
"""
from __future__ import annotations

import copy
import json
import sys

from .common import __version__


def _rank(rows):
    ranked = sorted([r for r in rows if r.get("score_agent") is not None], key=lambda r: -r["score_agent"])
    return {r["model"]: i for i, r in enumerate(ranked, 1)}


def _f(x, w=6):
    return f"{x:{w}.2f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"


def compare_table(old_rows, new_rows):
    """Lines of the old-vs-new table, best new rank first; models with neither score are left out."""
    ro, rn = _rank(old_rows), _rank(new_rows)
    old = {r["model"]: r for r in old_rows}
    lines = [f"  {'rank':>8}  {'model':<46} {'Agent score':>17}  {'quality':>17}  state"]
    for r in sorted(new_rows, key=lambda r: (rn.get(r["model"], 999), r["model"])):
        o = old.get(r["model"], {})
        if o.get("score_agent") is None and r.get("score_agent") is None:
            continue
        rank = f"{ro.get(r['model'], '-')!s:>3} -> {rn.get(r['model'], '-')!s:<3}"
        state = o.get("rank_state") or "-"
        if (r.get("rank_state") or "-") != state:
            state = f"{state} -> {r.get('rank_state') or '-'}"
        lines.append(f"  {rank:>8}  {r['model'][:46]:<46} {_f(o.get('score_agent'))} -> {_f(r.get('score_agent'))}  "
                     f"{_f(o.get('agent_quality'))} -> {_f(r.get('agent_quality'))}  {state}")
    moved = sum(ro.get(m) != rn.get(m) for m in set(ro) | set(rn))
    lines.append(f"  {moved} model(s) changed rank; {len(rn)} ranked now, {len(ro)} before.")
    return lines


def rescore_run(cfg, arg=None, write=True):
    from .reload import config_from_json, find_json, rows_from_json
    from .scoring import finalize
    src = find_json(cfg, arg)
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"[error] --rescore: could not read {src}: {e}")
    if not isinstance(doc, dict) or "models" not in doc:
        sys.exit(f"[error] --rescore: {src} is not a benchmark results file.")
    snap = doc.get("config") or {}
    new_settings = [k for k in ("agent_longctx_basis",) if k not in snap]
    config_from_json(cfg, doc)
    old_rows, rows = rows_from_json(doc), rows_from_json(doc)
    finalize(rows, cfg)
    run = doc.get("run") or {}
    run_id = run.get("id") or src.stem.replace("benchmark_", "")
    made_by = (doc.get("tool") or {}).get("version") or "?"
    print(f"Rescored run {run_id} (written by v{made_by}) with v{__version__} scoring.")
    if new_settings:
        print("  Not recorded by that run, taken from today's settings: "
              + ", ".join(f"{k.upper()}={getattr(cfg, k)}" for k in new_settings))
    basis = next((r.get("agent_basis") for r in rows if r.get("agent_basis")), None)
    if basis:
        print(f"  Agent basis now: {basis}")
    for line in compare_table(old_rows, rows):
        print(line)
    if not write:
        return rows
    from .reporting.dashboard import render_dashboard
    from .reporting.json_out import _clean, _model_entry
    out = copy.deepcopy(doc)
    out["models"] = [_clean(_model_entry(r)) for r in rows]
    out.setdefault("tool", {})["rescored_by"] = __version__
    out["config"] = {**snap, "agent_longctx_basis": cfg.agent_longctx_basis}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    jpath = cfg.output_dir / f"rescored_{run_id}.json"
    jpath.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    host = doc.get("host") or {}
    html = cfg.output_dir / f"benchmark_dashboard_{run_id}_rescored.html"
    render_dashboard(rows, cfg, html, host.get("cpu_power_source") or "none", host.get("hardware") or "",
                     partial=run.get("status") not in (None, "complete"))
    print(f"JSON:      {jpath}\nDashboard: {html}\n(The run's own files and the latest dashboard are unchanged.)")
    return rows
