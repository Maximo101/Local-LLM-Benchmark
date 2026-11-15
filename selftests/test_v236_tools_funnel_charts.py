"""v2.3.6: tool calling grows with the profile (32 core cases first, 96 in all), the gate counts tool calling's error
while more tool cases are to come, and the dashboard leaves early-stopped funnel models out of the accuracy charts."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.config import EVAL_SPECS, PROFILE_ROWS, TOOL_MIN_CASES, funnel_stages, rows_for
from llmbench.evals import funnel, toolcall
from llmbench.evals.datasets import sample_indices
from llmbench.evals.runner import _stage_ids
from llmbench.evals.toolcall_cases import CASES, TOOLS, build_items

from test_v23_agent import _row, cfg  # noqa: F401  (fixture + row builder)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "llmbench" / "reporting" / "dashboard.html"

# The v2.3 core set, in order. Its tools, prompts and system prompt are part of every cached answer's key: the
# digest below changes if any of them is edited (multiple-10b is the one deliberate change of v2.3.6).
CORE_IDS = ["simple-01", "simple-02", "simple-03", "simple-04", "simple-05", "simple-06", "simple-07", "simple-08",
            "simple-09", "simple-10", "multiple-01", "multiple-02", "multiple-03", "multiple-04", "multiple-05",
            "multiple-06", "multiple-07", "multiple-08", "multiple-09", "multiple-10b", "parallel-01", "parallel-02",
            "parallel-03", "parallel-04", "parallel-05", "parallel-06", "irrelevance-01", "irrelevance-02",
            "irrelevance-03", "irrelevance-04", "irrelevance-05", "irrelevance-06"]
CORE_DIGEST = "1b08f53178267922"      # the 31 unchanged v2.3.5 core items (computed from the v2.3.5 file)


def _digest(items):
    body = [{k: it[k] for k in ("id", "messages", "tools")} for it in items]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- the pool and its order
def test_core_cases_come_first_and_are_unchanged():
    assert [c["id"] for c in CASES[:TOOL_MIN_CASES]] == CORE_IDS
    assert len(CASES) == 96 and len({c["id"] for c in CASES}) == 96
    core = [it for it in build_items()[:TOOL_MIN_CASES] if it["id"] != "tool-multiple-10b"]
    assert _digest(core) == CORE_DIGEST


def test_rows_for_tool_calling_has_a_floor_and_grows():
    assert EVAL_SPECS["toolcall"]["min_rows"] == TOOL_MIN_CASES == 32 and not EVAL_SPECS["toolcall"].get("full")
    n = len(CASES)
    got = {p: (n if rows_for("toolcall", r) is None else min(n, rows_for("toolcall", r))) for p, r in PROFILE_ROWS.items()}
    assert got["test"] == got["quick"] == got["basic"] == 32
    assert got["moderate"] == 43 and got["standard"] == got["thorough"] == got["full"] == 96
    assert rows_for("mmlu_pro", 11) == 11 and rows_for("mmlu_pro", None) is None      # other datasets unchanged


def test_tool_cases_are_a_prefix_not_a_shuffle():
    for k in (32, 43, 96):
        assert sample_indices(96, k, 42, "toolcall") == list(range(k))
    assert sample_indices(500, 43, 42, "mbpp") != list(range(43))                     # datasets still shuffled


def test_funnel_stages_ask_the_core_first_then_the_new_cases():
    c = SimpleNamespace(eval_funnel_stages=(), eval_funnel_final="moderate", eval_seed=42,
                        eval_pools={"toolcall": (96, None)})
    items = [{**it, "_row": i} for i, it in enumerate(build_items())][:43]
    sets = _stage_ids(c, {"toolcall": items}, funnel_stages(c))["toolcall"]
    assert [len(s) for s in sets] == [32, 32, 43]
    assert sets[0] == {"tool-" + i for i in CORE_IDS} and sets[0] < sets[2]


def test_categories_are_mixed_in_every_block():
    cats = lambda a, b: {c["category"] for c in CASES[a:b]}
    assert cats(32, 43) == {"simple", "multiple", "parallel", "irrelevance", "missing", "chained"}
    assert cats(43, 96) == cats(32, 43)


# --------------------------------------------------------------------------- the cases grade as intended
def _reply_for(case):
    calls = [{"function": {"name": e["name"], "arguments": json.dumps({k: v[0] for k, v in e["args"].items()
                                                                       if v[0] is not None})}} for e in case["expect"]]
    return toolcall.encode_reply("" if calls else "A direct answer.", calls)


def test_every_case_accepts_its_own_expectation_and_rejects_the_opposite():
    items = {it["id"]: it for it in build_items()}
    for c in CASES:
        it = items["tool-" + c["id"]]
        ok, _, why = toolcall.grade_toolcall(it, _reply_for(c))
        assert ok, (c["id"], why)
        no_call = toolcall.grade_toolcall(it, toolcall.encode_reply("Answer.", []))[0]
        assert no_call == (not c["expect"]), c["id"]
        for e in c["expect"]:                                      # schema-consistent and in the offered tools
            assert e["name"] in c["tools"] and set(e["args"]) <= set(TOOLS[e["name"]]["function"]["parameters"]["properties"])


def test_missing_info_cases_fail_a_made_up_value():
    it = next(i for i in build_items() if i["id"] == "tool-missing-01")
    made_up = toolcall.encode_reply("", [{"function": {"name": "send_email", "arguments": json.dumps(
        {"to": "landlord@example.com", "subject": "Hot water", "body": "The hot water is broken."})}}])
    ok, _, why = toolcall.grade_toolcall(it, made_up)
    assert not ok and "needs none" in why


def test_chained_cases_carry_the_earlier_call_and_need_its_result():
    it = next(i for i in build_items() if i["id"] == "tool-chained-01")
    roles = [m["role"] for m in it["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    a, t = it["messages"][2], it["messages"][3]
    assert a["tool_calls"][0]["id"] == t["tool_call_id"] and len(t["tool_call_id"]) == 9 and t["tool_call_id"].isalnum()
    repeat = toolcall.encode_reply("", [{"function": {"name": "get_stock_price", "arguments": '{"ticker": "AAPL"}'}}])
    assert not toolcall.grade_toolcall(it, repeat)[0]                       # calling the first tool again is wrong
    wrong_amount = toolcall.encode_reply("", [{"function": {"name": "convert_currency", "arguments": json.dumps(
        {"amount": 1, "from_currency": "USD", "to_currency": "AUD"})}}])
    assert not toolcall.grade_toolcall(it, wrong_amount)[0]                 # the result's price must be used
    warm = next(i for i in build_items() if i["id"] == "tool-chained-03")
    assert warm["expect"] == [] and json.loads(warm["messages"][3]["content"])["temperature"] == 24


def test_multiple_10_was_replaced_by_an_explicit_request():
    ids = [c["id"] for c in CASES]
    assert "multiple-10" not in ids and ids.index("multiple-10b") == 19
    assert "translation tool" in CASES[19]["prompt"]


# --------------------------------------------------------------------------- the gate
def _rows(tool_n):
    return [_row("A-9B-Q4_K_M", mmlu=92, he=92, tools=95.0, lc={"64k": 100.0}, n=11, tool_n=tool_n),
            _row("B-9B-Q4_K_M", mmlu=80, he=80, tools=90.0, lc={"64k": 100.0}, n=11, tool_n=tool_n)]


@pytest.mark.parametrize("final,grows", [("moderate", True), ("standard", True), ("basic", False)])
def test_gate_counts_tool_error_while_more_tool_cases_are_to_come(cfg, final, grows):
    keys, a = ["mmlu_pro", "humaneval", "toolcall"], "A-9B-Q4_K_M"
    cfg.eval_pools = {"toolcall": (96, None)}
    cfg.eval_funnel_final = "basic"                      # 32 tool cases at every stage: the v2.3.3 rule
    red_v233 = funnel.composite_estimates_red(_rows(32), keys, cfg)[2][a]
    cfg.eval_funnel_final = final
    assert funnel.tool_cases_final(cfg) == {"moderate": 43, "standard": 96, "basic": 32}[final]
    red = funnel.composite_estimates_red(_rows(32), keys, cfg)[2][a]
    if grows:
        assert red > red_v233
    else:
        assert red == pytest.approx(red_v233)
    # once the model has answered every tool case the final asks, only the evals part's error is left
    done = funnel.tool_cases_final(cfg)
    assert funnel.composite_estimates_red(_rows(done), keys, cfg)[2][a] == pytest.approx(red_v233)


def test_gate_with_growing_tools_never_screens_out_more(cfg):
    """The widened bound only keeps more models (the safe side), never fewer."""
    keys = ["mmlu_pro", "humaneval", "toolcall"]
    rows = [_row(f"M{i}-9B-Q4_K_M", mmlu=95 - 3 * i, he=95 - 3 * i, tools=100 - 2 * i, lc={"64k": 100.0}, n=11,
                 tool_n=32) for i in range(8)]
    cfg.eval_pools = {"toolcall": (96, None)}
    cfg.eval_funnel_final = "basic"
    keep_old, _, _ = funnel.gate(rows, keys, leaders=2, cfg=cfg)
    cfg.eval_funnel_final = "standard"
    keep_new, _, _ = funnel.gate(rows, keys, leaders=2, cfg=cfg)
    assert keep_old <= keep_new


# --------------------------------------------------------------------------- estimate and settings page
def test_estimate_and_page_know_the_tool_counts(cfg):
    from llmbench import estimate
    from llmbench.ui import server
    cfg.eval_profile, cfg.eval_funnel_final = "funnel", "moderate"
    shape = estimate._stage_shape(cfg, 10)
    sh = estimate._key_shape(cfg, "toolcall", 43, shape)
    assert [round(f, 3) for f, _ in sh] == [round(32 / 43, 3), round(32 / 43, 3), 1.0]
    assert estimate._key_shape(cfg, "mmlu_pro", 43, shape) == shape
    g = server.guides({})
    per = {p["name"]: p["tests"]["toolcall"]["n"] for p in g["profiles"]}
    assert per["quick"] == 32 and per["moderate"] == 43 and per["standard"] == 96 and per["full"] == 96


# --------------------------------------------------------------------------- dashboard
def test_dashboard_leaves_early_stopped_models_out_of_the_accuracy_charts():
    t = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="evearly"' in t and "const evEarly" in t and "const atFinal" in t
    assert "evearly: v.evearly === true" in t                                   # hidden by default
    assert "EVR = VIEW.evearly ? OKR : OKR.filter(m=>!evEarly(m))" in t
    assert "const sp = EVR.filter(" in t                                         # quality vs speed too
    assert "ms.filter(m=>!evEarly(m)).map(m=>m.eval_avg_pct)" in t               # quant table eval average
    # long-context is asked in full at the first stage: that chart keeps every model
    assert "const lp = OKR.filter(m=>LCL.some(" in t


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_early_filter_logic_under_node(tmp_path):
    t = TEMPLATE.read_text(encoding="utf-8")
    start = t.index("const stageRows = s =>")
    end = t.index("const yTicksEarly")
    code = ("const EV=[{key:'mmlu_pro'}]; const evPct=(m,e)=>m['eval_'+e.key+'_pct'];"
            "const FUNNEL={stages:[['quick',11],['basic',24],['moderate',43]]}; const FUNNEL_FINAL='moderate';\n"
            + t[start:end] +
            "\nconst a={eval_stage:'moderate',eval_mmlu_pro_pct:80}, b={eval_stage:'quick',eval_mmlu_pro_pct:100,"
            "rank_state:'no_agent'}, c={eval_stage:'quick'}, d={eval_mmlu_pro_pct:70};\n"
            "console.log(JSON.stringify([evEarly(a), evEarly(b), evEarly(c), evEarly(d), earlyTip(b)]));")
    f = tmp_path / "t.js"
    f.write_text(code)
    out = json.loads(subprocess.run(["node", str(f)], capture_output=True, text=True, check=True).stdout)
    assert out[:4] == [False, True, False, False]
    assert "quick stage (11 questions per dataset, finalists 43)" in out[4] and "no Agent score" in out[4]


# --------------------------------------------------------------------------- end to end through the mock server
@pytest.fixture(scope="module")
def server():
    import mock_server
    s = mock_server.MockServer().start()
    yield s
    s.stop()


def test_funnel_run_asks_more_tool_cases_at_the_later_stage(monkeypatch, tmp_path, server, capsys):
    import mock_server
    from test_pipeline import ev, run_cli
    code, _, doc = run_cli(monkeypatch, tmp_path, server, ["--evals-only"], datasets=["gsm8k"], EVAL_TOOLCALL="true",
                           EVAL_PROFILE="funnel", EVAL_FUNNEL_STAGES="quick,moderate", EVAL_LEADERS="2",
                           EVAL_RESULT_CACHE="false")
    out = capsys.readouterr().out
    assert code == 0
    assert "Tool calling: 43 of 96 cases (built in)" in out
    assert "(tool calling 32 -> 43 cases)" in out and "tool calling's while more tool cases are to come" in out
    for m in doc["models"]:
        t = ev(m, "toolcall")
        assert t["n"] == 43 and m["evaluation"]["stage"] == "moderate"
        assert t["pct"] == pytest.approx(100 * 41 / 43)        # the mock's two deliberate mistakes are core cases
    tool_reqs = [r for r in mock_server.STATE["requests"] if r.get("tools")]
    first = tool_reqs[0]["model"]
    asked = [json.dumps(r["messages"], sort_keys=True) for r in tool_reqs if r["model"] == first]
    assert len(asked) == len(set(asked)) == 43                 # nested: no case asked twice
    assert any(m["role"] == "tool" for r in tool_reqs for m in r["messages"])     # chained cases went out
