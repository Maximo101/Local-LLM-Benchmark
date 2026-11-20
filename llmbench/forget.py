"""--forget-run STAMP (v2.4): remove one run's files from OUTPUT_DIR, e.g. a test run or one that went wrong.

Removed: every file of that run in OUTPUT_DIR whose name carries the stamp (benchmark_<stamp>.csv/.json/.log, its
dashboards, eval_details_<stamp>.jsonl, run_<stamp>.rows.json, rescored_<stamp>.json, pause files) and that run's
rows in benchmark_history_v2.csv (matched on model + timestamp from the run's own CSV; the old file is kept as .bak).
If results.json is that run, it is replaced by the newest remaining benchmark_<stamp>.json (or removed when none is
left).

Kept: the result cache (results/cache/). Its answers belong to the model fingerprint, not to a run, and another run
may be reusing them; use EVAL_RESULT_CACHE=refresh to ask again. The latest dashboard next to benchmark.py is not
touched.

Without --force it only lists what it would do.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys

_STAMP = re.compile(r"^\d{8}_\d{6}$")
HISTORY = "benchmark_history_v2.csv"


def run_files(out_dir, stamp):
    """Files of that run in OUTPUT_DIR (not in subfolders)."""
    return sorted(p for p in out_dir.iterdir() if p.is_file() and stamp in p.name and p.name != HISTORY)


def history_rows(out_dir, stamp):
    """(header, kept rows, removed count) of the history CSV without this run's rows; None when there is no history
    or no CSV of the run to match on."""
    hist, own = out_dir / HISTORY, out_dir / f"benchmark_{stamp}.csv"
    if not hist.exists() or not own.exists():
        return None
    with open(own, newline="", encoding="utf-8") as fh:
        keys = {(r.get("model"), r.get("timestamp")) for r in csv.DictReader(fh)}
    with open(hist, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        rows = list(rd)
    if not header or "model" not in header or "timestamp" not in header:
        return None
    mi, ti = header.index("model"), header.index("timestamp")
    kept = [r for r in rows if len(r) <= max(mi, ti) or (r[mi], r[ti]) not in keys]
    return header, kept, len(rows) - len(kept)


def _results_is(out_dir, stamp):
    p = out_dir / "results.json"
    try:
        return p.exists() and (json.loads(p.read_text(encoding="utf-8")).get("run") or {}).get("id") == stamp
    except (OSError, ValueError, AttributeError):
        return False


def forget_run(cfg, stamp, force=False):
    stamp = (stamp or "").strip()
    if not _STAMP.match(stamp):
        sys.exit(f"[error] --forget-run: {stamp!r} is not a run stamp (YYYYMMDD_HHMMSS, as in benchmark_<stamp>.json)")
    out = cfg.output_dir
    if not out.is_dir():
        sys.exit(f"[error] --forget-run: no output folder {out}")
    from .cli import lock_info
    try:
        st = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    if st.get("stamp") == stamp and st.get("state") == "running" and lock_info(cfg):
        sys.exit(f"[error] --forget-run: run {stamp} is still going. Stop it first.")
    files = run_files(out, stamp)
    hist = history_rows(out, stamp)
    is_latest = _results_is(out, stamp)
    if not files and not (hist and hist[2]) and not is_latest:
        print(f"--forget-run: nothing of run {stamp} in {out}.")
        return 0
    print(f"{'Removing' if force else 'Would remove'} run {stamp} from {out}:")
    for p in files:
        print(f"  {p.name}")
    if hist and hist[2]:
        print(f"  {hist[2]} row(s) of {HISTORY} (the old file is kept as {HISTORY}.bak)")
    newest = None
    if is_latest:
        rest = sorted((p for p in out.glob("benchmark_*.json") if stamp not in p.name), key=os.path.getmtime)
        newest = rest[-1] if rest else None
        print("  results.json is this run: " + (f"replaced by {newest.name}" if newest else "removed (no other run)"))
    print("  (the result cache is kept: its answers belong to the models, not to a run)")
    if not force:
        print("Nothing removed. Add --force to remove these.")
        return 0
    for p in files:
        p.unlink()
    if hist and hist[2]:
        path = out / HISTORY
        shutil.copy2(path, path.with_name(HISTORY + ".bak"))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(hist[0])
            w.writerows(hist[1])
    if is_latest:
        if newest:
            shutil.copy2(newest, out / "results.json")
        else:
            (out / "results.json").unlink(missing_ok=True)
    print(f"Run {stamp} removed.")
    return 0
