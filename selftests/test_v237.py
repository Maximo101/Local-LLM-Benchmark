"""v2.3.7: tokens per word without the chat template, stable long-context documents, cache reuse across thinking
budgets, llama.cpp build-change warnings, the missing-information reason text."""
import json

from llmbench import estimate, speed
from llmbench.evals import cache, longctx, toolcall
from llmbench.evals.toolcall_cases import build_items

TECH = "Explain quantum computing in 3 sentences."


def _runs(overhead, tpw=1.18, words=1200):
    pre = round(overhead + tpw * (words + speed.PREFILL_HEADER_WORDS))
    tech = round(overhead + tpw * len(TECH.split()))
    return {"prefill": [{"prompt_n": pre, "_words": words}] * 3, "tech": [{"prompt_n": tech}] * 3}


def test_template_overhead_no_longer_inflates_tokens_per_word():
    plain, padded = _runs(12), _runs(190)          # role markers only vs a template with a default system prompt
    a, b = speed.tokens_per_word(plain, TECH), speed.tokens_per_word(padded, TECH)
    assert abs(a - 1.18) < 0.01 and abs(b - 1.18) < 0.01
    old = speed.tokens_per_word(padded)            # no tech prompt: the v2.3.6 formula
    assert old > 1.3                               # what Cyber-Tiel read in the 2026-11-19 run (1.329)
    assert 170 <= speed.template_overhead(padded, b, TECH) <= 210


def test_tokens_per_word_falls_back_without_a_tech_count():
    r = _runs(12)
    r["tech"] = []
    assert abs(speed.tokens_per_word(r, TECH) - r["prefill"][0]["prompt_n"] / (1200 + 14)) < 1e-9
    assert speed.tokens_per_word({}, TECH) == 1.2


def test_haystack_figure_is_rounded_up_and_stable():
    assert longctx.haystack_tpw(1.172) == longctx.haystack_tpw(1.177) == 1.18
    assert longctx.haystack_tpw(1.18) == 1.18
    assert longctx.haystack_tpw(None) == 1.26
    for x in (1.001, 1.019, 1.333):
        assert x <= longctx.haystack_tpw(x) < x + 0.02


class _Cfg:
    eval_loop_abort = eval_salvage = eval_retry_on_length = eval_stream = True
    eval_budget_forcing = False
    eval_toolcall_strict = True
    eval_lcb_max_tests = 20
    eval_code_timeout = 10.0


def _rec(think, tokens, end="stop", **kw):
    return {"correct": True, "think_tokens": think, "tokens": tokens, "capped": False,
            "attempts": [{"mode": "think", "end": end, "tokens": tokens, "think": think}], **kw}


def test_budget_free_key_ignores_budget_and_max_tokens_only():
    item = {"id": "q1"}
    a = cache.budget_free_key(_Cfg, "mbpp", item, {"messages": [1], "max_tokens": 36864})
    b = cache.budget_free_key(_Cfg, "mbpp", item, {"messages": [1], "max_tokens": 20480})
    c = cache.budget_free_key(_Cfg, "mbpp", item, {"messages": [2], "max_tokens": 36864})
    assert a == b != c
    assert cache.question_key(_Cfg, "mbpp", item, {"messages": [1], "max_tokens": 36864}, 32768) != \
        cache.question_key(_Cfg, "mbpp", item, {"messages": [1], "max_tokens": 20480}, 16384)


def test_fits_budget_only_reuses_answers_no_limit_touched():
    assert cache.fits_budget(_rec(9000, 9500), 16384, 20480)
    assert not cache.fits_budget(_rec(17000, 17400), 16384, 20480)       # thought past the new budget
    assert not cache.fits_budget(_rec(9000, 21000), 16384, 20480)        # longer than the new max_tokens
    assert not cache.fits_budget(_rec(32769, 33000, end="budget"), 65536, 70000)   # hit the old cap
    assert not cache.fits_budget(_rec(100, 200, end="timeout"), 16384, 20480)
    assert not cache.fits_budget(_rec(100, 200, capped=True), 16384, 20480)
    two = _rec(100, 200)
    two["attempts"].append({"end": "stop"})                              # a retry happened: not reusable
    assert not cache.fits_budget(two, 16384, 20480)
    assert cache.fits_budget(_rec(5000, 5200), 0, None)                  # unlimited budget


def test_cache_finds_an_answer_asked_with_another_budget(tmp_path):
    c = cache.ResultCache(tmp_path)
    c.store("m", "fp", "q-old-budget", _rec(9000, 9500), akey="A")
    c.store("m", "fp", "q-old-budget-2", _rec(30000, 30500), akey="B")
    assert c.lookup("m", "fp", "q-new-budget") is None
    assert c.lookup_any_budget("m", "fp", "A", 16384, 20480)["think_tokens"] == 9000
    assert c.lookup_any_budget("m", "fp", "B", 16384, 20480) is None
    fresh = cache.ResultCache(tmp_path)                                  # read back from disk
    assert fresh.lookup_any_budget("m", "fp", "A", 16384, 20480) is not None
    (tmp_path / "old.jsonl").write_text(json.dumps({"qkey": "x", "rec": _rec(1, 2)}) + "\n")   # pre-2.3.7 line
    assert fresh.lookup_any_budget("m", "fp", "x", 16384, 20480) is None


def test_build_change_is_reported_per_build():
    prev = {"a": {"build": "b12316"}, "b": {"build": "b12316"}, "c": {"build": None}, "d": {"build": "b12407"}}
    rows = [{"model": m, "status": "ok", "build_info": "b12407"} for m in "abcd"] + \
           [{"model": "e", "status": "ok", "build_info": "b12407"}]
    ch = cache.build_changes(rows, prev)
    assert ch == {("b12316", "b12407"): ["a", "b"]}
    line = cache.build_change_lines(ch, 5)[0]
    assert "b12316 -> b12407" in line and "2 of 5" in line and "NOT be reused" in line
    assert cache.build_changes(rows, {}) == {}


def test_estimate_sees_stale_builds():
    hist = {"a": {"build_info": "b12316"}, "b": {"build_info": "b12407"}, "c": {}}
    build, stale = estimate.stale_builds(None, ["a", "b", "c", "z"], hist, build="b12407")
    assert build == "b12407" and stale == {"a": "b12316"}


def test_irrelevance_wording_is_kept():
    it = next(i for i in build_items() if i["id"].startswith("tool-irrelevance"))
    call = toolcall.encode_reply("", [{"function": {"name": it["tools"][0]["function"]["name"], "arguments": "{}"}}])
    ok, _, why = toolcall.grade_toolcall(it, call)
    assert not ok and "needs none" in why
