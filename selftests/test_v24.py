"""v2.4: a long-context basis that does not depend on which models reach the final, borderline long-context probes,
--rescore, --forget-run, --compare grading checks, --estimate funnel survival and resume."""
import csv
import json
import re

import pytest

from llmbench import agent, estimate, forget, rescore
from llmbench.compare import check
from llmbench.config import _lc_basis, load_config
from llmbench.evals import longctx
from llmbench.reload import rows_from_json
from llmbench.reporting.json_out import write_json
from llmbench.scoring import blank, finalize


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    import os
    for k in list(os.environ):
        if k.startswith(("EVAL_", "GPU_", "LLAMA_", "RUN_", "BUDGET_", "MIN_", "AGENT_", "VRAM_", "LONGCTX_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "results"))
    c = load_config()
    c.output_dir.mkdir(parents=True, exist_ok=True)
    return c


def _row(name, lc, ctx=131072, mmlu=90.0, he=90.0, tools=100.0, n=96):
    r = blank(name)
    r.update(tech_tps=120.0, logic_tps=120.0, vram_gb=13.6, vram_total_gb=16.0, ctx_size=ctx, agent_ttft_s=5.0,
             offload="none", status="ok")
    for k, p in (("mmlu_pro", mmlu), ("humaneval", he), ("toolcall", tools)):
        r.update({f"eval_{k}_pct": p, f"eval_{k}_n": n, f"eval_{k}_rows": n, f"eval_{k}_sec": 20.0,
                  f"eval_{k}_err": 0, f"eval_{k}_fallback": 0})
    for l, p in lc.items():
        r[f"longctx_{l}_pct"] = p
    return r


# --------------------------------------------------------------------------- the long-context basis
def test_basis_setting_is_parsed():
    assert _lc_basis("92160,65536") == "64k,90k"
    assert _lc_basis("90k, 64k, 64k") == "64k,90k"
    assert _lc_basis("COMMON") == "common" and _lc_basis("all") == "all" and _lc_basis("") == "64k,90k"
    with pytest.raises(SystemExit):
        _lc_basis("100k")


def test_a_short_context_finalist_no_longer_changes_everyone_else(cfg):
    """2026-11-16 vs -19: Bolt-E4B (64k) reaching the final took 90k out of every composite and moved Qwen3.6-MTP
    from #5 to #10 with identical answers. Now each model is judged on the basis lengths its context fits."""
    def run(with_short):
        a = _row("A-35B-A3B-Q4_K_M", {"64k": 100.0, "90k": 75.0, "128k": 50.0})
        b = _row("B-35B-A3B-Q4_K_M", {"64k": 100.0, "90k": 100.0, "128k": 100.0}, mmlu=88.0)
        rows = [a, b] + ([_row("C-4B-Q8_0", {"64k": 100.0}, ctx=65536)] if with_short else [])
        finalize(rows, cfg)
        return {r["model"]: r for r in rows}
    alone, crowd = run(False), run(True)
    for m in ("A-35B-A3B-Q4_K_M", "B-35B-A3B-Q4_K_M"):
        for k in ("agent_longctx_pct", "agent_composite_pct", "agent_quality", "score_agent"):
            assert alone[m][k] == pytest.approx(crowd[m][k]), (m, k)
    assert alone["A-35B-A3B-Q4_K_M"]["agent_longctx_pct"] == 87.5            # 64k and 90k; 128k is not in the basis
    assert crowd["C-4B-Q8_0"]["agent_longctx_pct"] == 100.0                   # 64k only: all its context fits
    assert "(64k, 90k; each model on those its context fits)" in crowd["C-4B-Q8_0"]["agent_basis"]


def test_common_and_all_basis(cfg):
    a = _row("A-35B-A3B-Q4_K_M", {"64k": 100.0, "90k": 75.0, "128k": 50.0})
    c = _row("C-4B-Q8_0", {"64k": 100.0}, ctx=65536)
    cfg.agent_longctx_basis = "common"
    finalize([a, c], cfg)
    assert a["agent_longctx_pct"] == 100.0 and "long-ctx 25% (64k)" in a["agent_basis"]
    cfg.agent_longctx_basis = "all"
    finalize([a, c], cfg)
    assert a["agent_longctx_pct"] == 75.0                                     # 64k, 90k and 128k


def test_more_retrievals_shrink_the_error(cfg):
    lengths = ["64k"]
    a = _row("A-35B-A3B-Q4_K_M", {"64k": 75.0})
    w = {"longctx": 1.0}
    _, se8 = agent.composite(a, w, [], False, lengths, 8)
    a["longctx_64k_n"] = 16
    v, se16 = agent.composite(a, w, [], False, lengths, 8)
    assert v == 75.0 and se16 < se8


# --------------------------------------------------------------------------- borderline probes
def test_borderline_is_within_one_code_of_the_pass_mark():
    assert [h for h in range(9) if longctx.borderline(h, 8, 75)] == [5, 6]
    assert [h for h in range(17) if longctx.borderline(h, 16, 75)] == [11, 12]
    assert not longctx.borderline(5, 8, 0) and not longctx.borderline(0, 0, 75)


def _fake_asker(plan):
    """A model that gives the right code for the first plan[i] vaults of question i."""
    calls = []

    def ask(cfg, session, model, prompt, timeout, max_tokens=None):
        codes = dict(re.findall(r"vault (\w+) is (\d{6})", prompt))
        names = re.search(r"these vaults: (.*?)\. Reply", prompt).group(1).split(", ")
        right = plan[len(calls)]
        calls.append(names)
        text = "\n".join(f"Vault {n}: {codes[n] if i < right else '000000'}" for i, n in enumerate(names))
        return text, 64000, {"prompt_per_second": 2000.0, "cache_n": 0 if len(calls) == 1 else 63000}
    return ask, calls


def test_a_borderline_length_gets_more_questions(cfg, monkeypatch):
    cfg.longctx_lengths = (65536,)
    ask, calls = _fake_asker([3, 2, 4, 4])                        # 5/8 -> borderline -> 8 more, all right
    monkeypatch.setattr(longctx, "_ask", ask)
    r = {"model": "M", "kv_type": "q8_0", "tok_per_word": 1.18}
    longctx.run_longctx(cfg, None, r, 131072)
    assert len(calls) == 4 and len({n for c in calls for n in c}) == 16  # 16 different vaults
    assert r["longctx_64k_n"] == 16 and r["longctx_64k_pct"] == pytest.approx(100 * 13 / 16)
    assert "within one code" in r["longctx_note"]


def test_a_clear_result_is_not_probed(cfg, monkeypatch):
    cfg.longctx_lengths = (65536,)
    for plan, pct in (([4, 3], 87.5), ([2, 2], 50.0)):
        ask, calls = _fake_asker(plan + [4, 4])
        monkeypatch.setattr(longctx, "_ask", ask)
        r = {"model": "M", "kv_type": "q8_0", "tok_per_word": 1.18}
        longctx.run_longctx(cfg, None, r, 131072)
        assert len(calls) == 2 and r["longctx_64k_n"] == 8 and r["longctx_64k_pct"] == pct
    cfg.longctx_borderline_trials = 0                             # off
    ask, calls = _fake_asker([3, 2])
    monkeypatch.setattr(longctx, "_ask", ask)
    r = {"model": "M", "kv_type": "q8_0", "tok_per_word": 1.18}
    longctx.run_longctx(cfg, None, r, 131072)
    assert len(calls) == 2 and r["longctx_64k_pct"] == 62.5


# --------------------------------------------------------------------------- --rescore
def _write_run(cfg, rows, stamp="20261116_090118"):
    finalize(rows, cfg)
    write_json(rows, cfg, {"stamp": stamp, "started": 1.0, "finished": 2.0, "status": "complete"}, stamp)
    return cfg.output_dir / f"benchmark_{stamp}.json"


def test_rescore_writes_new_scores_and_leaves_the_run_alone(cfg, capsys):
    cfg.agent_longctx_basis = "common"                            # the run was scored with the v2.3 rule
    rows = [_row("A-35B-A3B-Q4_K_M", {"64k": 100.0, "90k": 75.0}), _row("B-35B-A3B-Q4_K_M", {"64k": 100.0, "90k": 100.0}),
            _row("C-4B-Q8_0", {"64k": 100.0}, ctx=65536)]
    path = _write_run(cfg, rows)
    doc = json.loads(path.read_text())
    doc["config"].pop("agent_longctx_basis")                      # as a pre-v2.4 run
    path.write_text(json.dumps(doc))
    before = path.read_text()
    cfg.agent_longctx_basis = "64k,90k"
    new = {r["model"]: r for r in rescore.rescore_run(cfg, str(path))}
    out = capsys.readouterr().out
    assert "AGENT_LONGCTX_BASIS=64k,90k" in out and "changed rank" in out
    assert path.read_text() == before and not (cfg.output_dir / "results.json").read_text() == ""
    assert new["A-35B-A3B-Q4_K_M"]["agent_longctx_pct"] == 87.5
    again = {r["model"]: r for r in rows_from_json(json.loads((cfg.output_dir / "rescored_20261116_090118.json").read_text()))}
    assert again["A-35B-A3B-Q4_K_M"]["score_agent"] == pytest.approx(new["A-35B-A3B-Q4_K_M"]["score_agent"])
    assert (cfg.output_dir / "benchmark_dashboard_20261116_090118_rescored.html").exists()


def test_compare_table_marks_rank_moves():
    old = [{"model": "a", "score_agent": 90.0, "agent_quality": 99.0, "rank_state": "ranked"},
           {"model": "b", "score_agent": 89.0, "agent_quality": 98.0, "rank_state": "ranked"}]
    new = [{"model": "a", "score_agent": 88.0, "agent_quality": 97.0, "rank_state": "ranked"},
           {"model": "b", "score_agent": 89.0, "agent_quality": 98.0, "rank_state": "ranked"}]
    lines = rescore.compare_table(old, new)
    assert "2 model(s) changed rank" in lines[-1] and lines[1].lstrip().startswith("2 -> 1")


# --------------------------------------------------------------------------- --forget-run
def test_forget_run_lists_then_removes(cfg, capsys):
    rows = [_row("A-35B-A3B-Q4_K_M", {"64k": 100.0})]
    keep = _write_run(cfg, [dict(r) for r in rows], "20261115_154344")
    gone = _write_run(cfg, rows, "20261116_090118")
    out = cfg.output_dir
    for name in ("benchmark_20261116_090118.log", "eval_details_20261116_090118.jsonl", "run_20261116_090118.rows.json",
                 "benchmark_dashboard_20261116_090118.html"):
        (out / name).write_text("x")
    with open(out / "benchmark_20261116_090118.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["timestamp", "model"], ["2026-11-16 09:01:32", "A"]])
    with open(out / forget.HISTORY, "w", newline="") as fh:
        csv.writer(fh).writerows([["timestamp", "model"], ["2026-11-15 15:43:44", "A"], ["2026-11-16 09:01:32", "A"]])
    forget.forget_run(cfg, "20261116_090118")
    assert "Would remove" in capsys.readouterr().out and gone.exists()
    forget.forget_run(cfg, "20261116_090118", force=True)
    assert not gone.exists() and not (out / "eval_details_20261116_090118.jsonl").exists() and keep.exists()
    assert json.loads((out / "results.json").read_text())["run"]["id"] == "20261115_154344"
    assert list(csv.reader(open(out / forget.HISTORY))) == [["timestamp", "model"], ["2026-11-15 15:43:44", "A"]]
    assert (out / (forget.HISTORY + ".bak")).exists()
    with pytest.raises(SystemExit):
        forget.forget_run(cfg, "latest")


# --------------------------------------------------------------------------- --compare, --estimate
def test_compare_refuses_different_grading_and_notes_longctx():
    base = {"evaluation": {"datasets": {"mbpp": {"rows": 96}}, "long_context": {"trials": 2, "keys": 4}},
            "config": {"eval_code_timeout": 10.0}, "tool": {"version": "2.4.0"}}
    other = json.loads(json.dumps(base))
    other["config"]["eval_code_timeout"] = 30.0
    other["evaluation"]["long_context"]["trials"] = 4
    problems, notes = check([base, other], ["A", "B"])
    assert any("code-test timeout" in p for p in problems)
    assert any("LONGCTX_TRIALS 2 vs 4" in n for n in notes)
    old = json.loads(json.dumps(base))
    old["config"] = {}                                            # an older run that did not record it: not refused
    assert not check([base, old], ["A", "B"])[0]


def test_estimate_learns_funnel_survival(cfg):
    assert estimate.observed_survival(cfg) == (estimate.FUNNEL_SURVIVE, 0)
    gates = [{"kept": i % 10 != 0} for i in range(20)]            # 18 of 20 kept
    (cfg.output_dir / "benchmark_20261119_115908.json").write_text(
        json.dumps({"evaluation": {"funnel": {"gates": gates}}}))
    share, runs = estimate.observed_survival(cfg)
    assert share == pytest.approx(0.9) and runs == 1
