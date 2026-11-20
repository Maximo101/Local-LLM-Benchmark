"""v2.4.1: --diff-runs, the multi-run diff and its HTML / JSON report.

Everything here is mocked: synthetic runs from run_factory (the real json_out writes them) and, for the end-to-end
test, two evals-only runs against the mock llama-swap server. No GPU, llama-server, Docker or Hugging Face."""
import json
import re
import shutil
import subprocess
import sys
import time

import pytest

import run_factory as F
from llmbench import rundiff as rd
from llmbench.config import load_config


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


def _model(diff, name):
    return next(m for m in diff["models"] if m["model"] == name)


# --------------------------------------------------------------------------------------------- statistics
def test_mcnemar_exact_values():
    assert rd.mcnemar_p(0, 0) == 1.0
    assert rd.mcnemar_p(5, 5) == 1.0
    assert rd.mcnemar_p(0, 10) == pytest.approx(2 / 1024)
    assert rd.mcnemar_p(10, 0) == rd.mcnemar_p(0, 10)                      # two-sided, symmetric
    assert rd.mcnemar_p(1, 6) == pytest.approx(2 * 8 / 128)                 # (C(7,0) + C(7,1)) / 2^7, doubled
    assert rd.mcnemar_p(2, 3) > 0.05


def test_agresti_caffo_interval():
    lo, hi = rd.agresti_caffo(70.0, 96, 70.0, 96)
    assert lo < 0 < hi and lo == pytest.approx(-hi, abs=1e-9)
    lo, hi = rd.agresti_caffo(50.0, 200, 80.0, 200)                          # 30 points over 200 questions: real
    assert lo > 15 and hi < 40
    lo, hi = rd.agresti_caffo(70.0, 50, 76.0, 50)                            # 6 points over 50: noise
    assert lo < 0 < hi
    lo, hi = rd.agresti_caffo(0.0, 8, 100.0, 8)                              # no division by zero at 0% / 100%
    assert lo > 0
    assert rd.agresti_caffo(None, 8, 50.0, 8) is None and rd.agresti_caffo(50.0, 0, 50.0, 8) is None


def test_kendall_tau_b():
    assert rd.kendall_tau([3, 2, 1], [30, 20, 10]) == 1.0
    assert rd.kendall_tau([3, 2, 1], [10, 20, 30]) == -1.0
    assert rd.kendall_tau([1], [1]) is None
    t = rd.kendall_tau([1, 2, 3, 4], [1, 2, 2, 4])                           # a tie on one side
    assert 0.9 < t < 1.0
    assert rd.kendall_tau([1, 1], [2, 2]) is None                            # all tied: undefined


def test_metric_verdicts_respect_direction_and_noise():
    v = rd.metric_verdict
    assert v("tps", 100, 104, +1) == rd.SAME                                 # 4% < 5%
    assert v("tps", 100, 94, +1) == rd.WORSE
    assert v("tps", 100, 94, +1, spread=4.0) == rd.SAME                      # 2 x 4% spread = 8% noise
    assert v("time", 10.0, 8.0, -1) == rd.BETTER                             # lower is better
    assert v("vram", 13.60, 13.65, -1) == rd.SAME and v("vram", 13.6, 14.0, -1) == rd.WORSE
    assert v("score", 80.0, 80.4, +1) == rd.SAME and v("score", 80.0, 81.0, +1) == rd.BETTER
    assert v("power", 200, 215, -1) == rd.WORSE
    assert v("tps", None, 100, +1) is None and v("tps", float("nan"), 100, +1) is None


# --------------------------------------------------------------------------------------------- the diff
def test_diff_finds_the_planted_changes(cfg, capsys):
    pa, pb, a, b = F.two_runs(cfg)
    hp, jp, d = rd.diff_runs(cfg, [str(pa), str(pb)])
    m0, m1, m2, gone = a[0]["model"], a[1]["model"], a[2]["model"], a[5]["model"]
    g = _model(d, m0)["datasets"]["gsm8k"]
    assert g["test"] == "paired" and g["paired"]["c"] == 20 and g["paired"]["b"] == 0 and g["verdict"] == rd.BETTER
    assert g["delta"] == pytest.approx(40.0)
    mm = _model(d, m1)["datasets"]["mmlu_pro"]
    assert mm["test"] == "paired" and mm["paired"]["b"] == 2 and mm["verdict"] == rd.SAME     # 2 flips: noise
    assert _model(d, m0)["metrics"]["avg_tps"]["verdict"] == rd.WORSE
    assert _model(d, m0)["metrics"]["avg_tps"]["pct"] == pytest.approx(-20.0, abs=0.01)
    assert "ctx 65536 -> 131072" in _model(d, m2)["launch_changes"]
    assert any(c.startswith("llama.cpp build b12316 -> b12407") for c in _model(d, m0)["launch_changes"])
    s = d["summary"]
    assert s["new"] == ["New-14B-Q4_K_M.gguf"] and s["dropped"] == [gone] and s["common"] == 5
    assert s["accuracy_better"] == 1 and s["accuracy_worse"] == 0 and s["speed_worse"] >= 1
    assert _model(d, gone)["present"] == [True, False] and _model(d, "New-14B-Q4_K_M.gguf")["present"] == [False, True]
    assert d["problems"] == []
    assert hp.name.startswith("run_diff_") and jp.is_file()
    assert json.loads(jp.read_text())["summary"] == json.loads(json.dumps(s))
    out = capsys.readouterr().out
    assert "Run diff: 16 Nov 09:01 -> 19 Nov 11:59" in out and "Report:" in out


def test_without_details_the_unpaired_test_is_used_and_noted(cfg):
    pa, pb, a, _ = F.two_runs(cfg, details=False)
    _, _, d = rd.diff_runs(cfg, [str(pa), str(pb)])
    g = _model(d, a[0]["model"])["datasets"]["gsm8k"]
    assert g["test"] == "unpaired" and g["verdict"] == rd.BETTER and g["ci"][0] > 0      # +40 points over 50: real
    assert any("eval_details not found" in n for n in d["notes"])


def test_details_found_next_to_the_json_when_the_recorded_path_is_elsewhere(cfg, tmp_path):
    """The main PC reads the server's results over SMB: the recorded /mnt/user/... path does not exist there."""
    pa, pb, _, _ = F.two_runs(cfg)
    for p in (pa, pb):
        doc = json.loads(p.read_text())
        doc["artifacts"]["eval_details"] = "/mnt/user/scripts/llama/results/" + \
            p.name.replace("benchmark_", "eval_details_").replace(".json", ".jsonl")
        p.write_text(json.dumps(doc))
    _, _, d = rd.diff_runs(cfg, [str(pa), str(pb)])
    assert all(r["details"] for r in d["runs"])


def test_a_run_against_itself_changes_nothing(cfg):
    pa, _, _, _ = F.two_runs(cfg)
    twin = pa.with_name("benchmark_20261116_090119.json")
    shutil.copy(pa, twin)
    shutil.copy(pa.with_name("eval_details_20261116_090118.jsonl"), pa.with_name("eval_details_20261116_090119.jsonl"))
    doc = json.loads(twin.read_text())
    doc["run"]["id"], doc["artifacts"]["eval_details"] = "20261116_090119", ""
    twin.write_text(json.dumps(doc))
    _, _, d = rd.diff_runs(cfg, [str(pa), str(twin)])
    for m in d["models"]:
        for part in list(m["metrics"].values()) + list(m["datasets"].values()) + list(m["longctx"].values()):
            assert part["verdict"] in (rd.SAME, None), (m["model"], part)
        assert m["launch_changes"] == []
    assert d["rank_agreement"][0]["tau"] == 1.0
    assert d["summary"]["accuracy_better"] == d["summary"]["accuracy_worse"] == 0
    assert d["config_changes"] == [] and d["problems"] == []


def test_three_runs_draw_a_trend_and_compare_first_with_last(cfg):
    pa, pb, _, b = F.two_runs(cfg)
    pc = F.write_run(cfg, b, "20261120_080000")
    hp, _, d = rd.diff_runs(cfg, [str(pa), str(pb), str(pc)])
    assert len(d["runs"]) == 3 and len(d["rank_agreement"]) == 2
    assert d["rank_agreement"][1]["tau"] == 1.0                               # B and C are the same rows
    html = hp.read_text()
    assert html.count('<svg class="spark"') >= 3
    assert d["runs"][2]["details"] is False                                   # C has none: unpaired, noted
    assert any("eval_details not found for 20 Nov 08:00" in n for n in d["notes"])


def test_different_eval_settings_are_flagged_not_refused(cfg):
    a = F.base_rows(cfg)
    pa = F.write_run(cfg, a, "20261116_090118")
    pb = F.write_run(cfg, a, "20261119_115908", eval_seed=99, eval_code_timeout=99.0)
    hp, _, d = rd.diff_runs(cfg, [str(pa), str(pb)])
    assert any("eval seed" in p for p in d["problems"])
    assert any("code-test timeout" in p for p in d["problems"])
    keys = {c["key"] for c in d["config_changes"]}
    assert {"eval_seed", "eval_code_timeout"} <= keys and "output_dir" not in keys
    assert "Not like-for-like" in hp.read_text()


def test_scores_from_other_versions_are_noted_and_rescore_puts_them_on_one_footing(cfg):
    a = F.base_rows(cfg)
    pa = F.write_run(cfg, a, "20261116_090118", version="2.3.6")
    pb = F.write_run(cfg, a, "20261119_115908")
    _, _, d = rd.diff_runs(cfg, [str(pa), str(pb)])
    assert any("different versions" in n and "--diff-rescore" in n for n in d["notes"])
    _, _, d2 = rd.diff_runs(cfg, [str(pa), str(pb)], rescore=True)
    assert not any("--diff-rescore" in n for n in d2["notes"])
    assert d2["scores"].startswith("rescored with v") and {r["scored_by"] for r in d2["runs"]} == {rd.__version__}
    for m in d2["models"]:                                                    # same rows, same scoring: no change
        assert m["metrics"]["score_agent"]["verdict"] in (rd.SAME, None)


def test_rescore_does_not_touch_the_run_files(cfg):
    pa, pb, _, _ = F.two_runs(cfg)
    before = {p: p.read_bytes() for p in cfg.output_dir.iterdir()}
    rd.diff_runs(cfg, [str(pa), str(pb)], rescore=True)
    after = {p: p.read_bytes() for p in cfg.output_dir.iterdir() if p in before}
    assert after == before
    new = {p.name for p in cfg.output_dir.iterdir()} - {p.name for p in before}
    assert all(n.startswith("run_diff_") for n in new) and len(new) == 2


# --------------------------------------------------------------------------------------------- choosing runs
def test_run_selection(cfg, monkeypatch):
    pa, pb, _, b = F.two_runs(cfg)
    pc = F.write_run(cfg, b, "20261120_080000")
    demo = json.loads(pc.read_text())
    demo["run"]["mode"], demo["run"]["id"] = "demo", "20261121_000000"
    (cfg.output_dir / "benchmark_20261121_000000.json").write_text(json.dumps(demo))      # never picked
    assert rd.resolve_runs(cfg, []) == [pb, pc]
    assert rd.resolve_runs(cfg, ["3"]) == [pa, pb, pc]
    assert rd.resolve_runs(cfg, ["20261119_115908", "20261116_090118"]) == [pb, pa]       # given order kept
    (cfg.output_dir / "sub").mkdir()
    shutil.copy(pa, cfg.output_dir / "sub" / "results.json")
    assert rd.resolve_runs(cfg, [str(cfg.output_dir / "sub"), str(pb)])[0].name == "results.json"
    for bad in (["1"], [str(pa)], [str(pa), str(pa)], ["20990101_000000", str(pa)]):
        with pytest.raises(SystemExit):
            rd.resolve_runs(cfg, bad)


def test_a_rescored_file_is_accepted_by_stamp(cfg):
    from llmbench.rescore import rescore_run
    pa, pb, _, _ = F.two_runs(cfg)
    rescore_run(cfg, str(pa))
    pa.unlink()
    assert rd.resolve_runs(cfg, ["20261116_090118", "20261119_115908"])[0].name == "rescored_20261116_090118.json"
    _, _, d = rd.diff_runs(cfg, ["20261116_090118", "20261119_115908"])
    assert d["runs"][0]["label"].endswith("(rescored)")


def test_too_few_runs_in_the_output_folder(cfg):
    F.write_run(cfg, F.base_rows(cfg), "20261116_090118")
    with pytest.raises(SystemExit, match="two are needed"):
        rd.resolve_runs(cfg, [])


# --------------------------------------------------------------------------------------------- the page
def test_page_escapes_model_names_and_never_shows_question_text(cfg):
    a = F.base_rows(cfg)
    evil = '<script>alert("x")</script>-7B-Q4_K_M.gguf'
    a[0]["model"] = evil
    da = F.verdicts(a)
    pa = F.write_run(cfg, a, "20261116_090118", da)
    pb = F.write_run(cfg, a, "20261119_115908", da)
    hp, jp, _ = rd.diff_runs(cfg, [str(pa), str(pb)])
    page, js = hp.read_text(), jp.read_text()
    assert evil not in page and "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in page
    assert page.count("<script>") == 1                                       # only the page's own
    assert "Content-Security-Policy" in page and "http://" not in page and "https://" not in page
    assert F.SECRET not in page and F.SECRET not in js                       # verdicts only, never question text


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not on PATH")
def test_page_script_parses(cfg, tmp_path):
    pa, pb, _, _ = F.two_runs(cfg)
    hp, _, _ = rd.diff_runs(cfg, [str(pa), str(pb)])
    script = re.search(r"<script>(.*?)</script>", hp.read_text(), re.S).group(1)
    f = tmp_path / "page.js"
    f.write_text(script)
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_table_row_has_one_cell_per_header(cfg):
    pa, pb, _, b = F.two_runs(cfg)
    pc = F.write_run(cfg, b, "20261120_080000")
    hp, _, _ = rd.diff_runs(cfg, [str(pa), str(pb), str(pc)])
    for table in re.findall(r"<table.*?</table>", hp.read_text(), re.S):
        heads = len(re.findall(r"<th[ >]", table.split("</thead>")[0]))
        for row in re.findall(r"<tr[ >].*?</tr>", table.split("<tbody>")[1], re.S):
            assert len(re.findall(r"<td[ >]", row)) == heads, row[:120]


# --------------------------------------------------------------------------------------------- command line
def _main(monkeypatch, argv):
    from llmbench.cli import main
    monkeypatch.setattr(sys, "argv", ["benchmark.py"] + argv)
    try:
        main()
        return 0
    except SystemExit as e:
        return e.code


def test_cli_flags(cfg, monkeypatch, capsys):
    pa, pb, _, _ = F.two_runs(cfg)
    assert _main(monkeypatch, ["--diff-runs"]) == 0
    assert "Run diff: 16 Nov 09:01 -> 19 Nov 11:59" in capsys.readouterr().out
    time.sleep(1.1)                                                           # run_diff_<stamp> is per second
    assert _main(monkeypatch, ["--diff-runs", str(pa), str(pb), "--diff-rescore"]) == 0
    assert "rescored with v" in capsys.readouterr().out
    assert len(list(cfg.output_dir.glob("run_diff_*.html"))) == 2
    assert _main(monkeypatch, ["--diff-rescore"]) == 2                        # argparse error
    assert "goes with --diff-runs" in capsys.readouterr().err


def test_forget_run_does_not_take_diff_reports(cfg):
    from llmbench.forget import run_files
    pa, pb, _, _ = F.two_runs(cfg)
    rd.diff_runs(cfg, [str(pa), str(pb)])
    assert not any(p.name.startswith("run_diff_") for p in run_files(cfg.output_dir, "20261116_090118"))


# --------------------------------------------------------------------------------------------- end to end (mock server)
from test_pipeline import run_cli  # noqa: E402
from test_v21 import M1, M2, server  # noqa: E402,F401  (fixture)


def test_two_real_runs_against_the_mock_server(monkeypatch, tmp_path, server):
    """Two evals-only runs through the whole pipeline, then --diff-runs on what they wrote. The second run answers
    from the result cache, so every verdict is the same: nothing may be reported as a change."""
    code, _, d1 = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"])
    assert code == 0
    time.sleep(1.1)
    code, _, d2 = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k", "gpqa"])
    assert code == 0 and d1["run"]["id"] != d2["run"]["id"]
    time.sleep(1.1)
    code, _, _ = run_cli(monkeypatch, tmp_path, server, ["--diff-runs"])
    assert code == 0
    out = tmp_path / "results"
    jp = sorted(out.glob("run_diff_*.json"))[-1]
    d = json.loads(jp.read_text())
    assert [r["id"] for r in d["runs"]] == [d1["run"]["id"], d2["run"]["id"]]
    assert all(r["details"] for r in d["runs"])
    names = {m["model"] for m in d["models"]}
    assert {M1, M2} <= names
    for m in d["models"]:
        for k, ent in m["datasets"].items():
            assert ent["test"] == "paired" or ent["n"][0] < 10, (m["model"], k)
            assert ent["verdict"] == rd.SAME, (m["model"], k, ent)
    page = sorted(out.glob("run_diff_*.html"))[-1].read_text()
    gpqa_text = [json.loads(line).get("response") for line in
                 open(out / f"eval_details_{d1['run']['id']}.jsonl", encoding="utf-8")]
    for t in filter(None, gpqa_text):
        if len(t) > 12:
            assert t not in page                                              # no model reply reaches the report
