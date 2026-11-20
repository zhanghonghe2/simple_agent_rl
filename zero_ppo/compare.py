#!/usr/bin/env python3
"""Put several eval reports side by side, so the RL gain is separable from SFT's.

Three checkpoints matter here and they answer different questions:

    base   Qwen2.5-Math-1.5B-Instruct   what the pretrained model can do
    sft    the SFT model                what the tool protocol taught it
    rl     the RL checkpoint            what reinforcement added ON TOP of SFT

The RL delta must be measured against SFT, not against base - comparing RL to base
credits SFT's work to RL. Differences are reported in percentage points.

A caveat this script prints rather than hides: with 458 problems one point is
about 4.6 problems, so a gap of a point or two is inside sampling noise. The
standard error of an accuracy is sqrt(p(1-p)/n); gaps smaller than ~2 SE are
marked as not clearly separated.

    python3 -m zero_ppo.compare --self-test
    python3 -m zero_ppo.compare base=runs/eval_base sft=runs/eval_sft rl=runs/eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_report(path: str) -> dict:
    """Accept a report.json or the directory containing it."""
    if os.path.isdir(path):
        path = os.path.join(path, "report.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def se(p: float, n: int) -> float:
    """Standard error of a proportion, in the same units as p."""
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def comparable(reports: dict) -> list[str]:
    """Config fields that must match for the numbers to mean the same thing."""
    keys = ("pool", "samples", "temperature", "seed", "top_p", "max_turns",
            "max_model_len", "max_new", "greedy", "judge_api", "judge_model")
    cfgs = {name: r.get("config", {}) for name, r in reports.items()}
    warnings = []
    for k in keys:
        vals = {name: c.get(k) for name, c in cfgs.items()}
        if len({json.dumps(v, sort_keys=True) for v in vals.values()}) > 1:
            warnings.append(f"{k} 不一致: " +
                            ", ".join(f"{n}={v}" for n, v in vals.items()))
    # judge_api records the intent; judge_used records whether arbitration really
    # ran. A run that asked for the API but fell back to rules would otherwise
    # compare its loose numbers against a different standard.
    used = {name: c.get("judge_used") for name, c in cfgs.items()}
    if any(v is not None for v in used.values()):
        if len(set(used.values())) > 1:
            warnings.append("宽口径判分口径不一致（部分未真正调用 API）: " +
                            ", ".join(f"{n}={v}" for n, v in used.items()) +
                            " —— 宽口径数字不可直接比较")
        asked = {name: c.get("judge_api") for name, c in cfgs.items()}
        for name, c in cfgs.items():
            if c.get("judge_api") and c.get("judge_used") is False:
                warnings.append(f"{name} 请求了 API 判分但实际未执行，宽口径退化为纯规则")
        del asked
    # a different problem count means a different test set, not a weaker model
    ns = {name: r["overall"].get("problems") for name, r in reports.items()}
    if len(set(ns.values())) > 1:
        warnings.append(f"题数不一致: {ns} —— 不是同一测试集，不可直接比较")
    return warnings


METRICS = [
    ("greedy_strict", "贪心·严格"),
    ("greedy_loose", "贪心·宽口径"),
    ("pass@1_strict", "pass@1·严格"),
    ("pass@1_loose", "pass@1·宽口径"),
    ("pass@8_strict", "pass@8·严格"),
    ("pass@8_loose", "pass@8·宽口径"),
]


def build_table(reports: dict, order: list[str]) -> list[dict]:
    rows = []
    for key, label in METRICS:
        vals = {n: reports[n]["overall"].get(key) for n in order}
        if all(v is None for v in vals.values()):
            continue
        rows.append({"metric": label, "key": key, "values": vals})
    return rows


def delta_row(row: dict, a: str, b: str, n: int) -> str:
    """b - a in percentage points, flagged when inside noise."""
    va, vb = row["values"].get(a), row["values"].get(b)
    if va is None or vb is None:
        return "     -"
    d = (vb - va) * 100
    # SE of a difference of two independent proportions
    s = math.sqrt(se(va, n) ** 2 + se(vb, n) ** 2) * 100
    mark = "" if abs(d) > 2 * s else " ~"      # ~ = not clearly separated
    return f"{d:+6.2f}{mark}"


def print_table(reports: dict, order: list[str]) -> None:
    n = min(r["overall"].get("problems", 0) for r in reports.values())
    warns = comparable(reports)
    print("\n" + "=" * 78)
    print(f"  模型对比 · {n} 题")
    print("=" * 78)
    head = f"  {'指标':<18}" + "".join(f"{x:>12}" for x in order)
    print(head)
    print("-" * 78)
    for row in build_table(reports, order):
        line = f"  {row['metric']:<18}"
        for name in order:
            v = row["values"].get(name)
            line += f"{v*100:>11.2f}%" if v is not None else f"{'-':>12}"
        print(line)

    # deltas: RL vs SFT is the one that matters, base vs SFT shows SFT's own gain
    pairs = [(a, b) for a, b in zip(order, order[1:])]
    if len(order) >= 3:
        pairs.append((order[0], order[-1]))
    if pairs:
        print("-" * 78)
        print("  差值（百分点，~ 表示落在采样噪声内，未明显区分）")
        for a, b in pairs:
            print(f"\n  {b} − {a}")
            for row in build_table(reports, order):
                print(f"    {row['metric']:<18}{delta_row(row, a, b, n):>12}")

    print("-" * 78)
    print(f"  {'诊断':<18}" + "".join(f"{x:>12}" for x in order))
    for key, label in (("boxed_rate", "出框率"), ("avg_turns", "平均轮数"),
                       ("tool_ok_per_traj", "工具成功数"),
                       ("cut_context", "上下文耗尽"), ("cut_budget", "生成压缩")):
        line = f"  {label:<18}"
        for name in order:
            v = reports[name]["overall"].get(key)
            if v is None:
                line += f"{'-':>12}"
            elif key in ("boxed_rate", "cut_context", "cut_budget"):
                line += f"{v*100:>11.1f}%"
            else:
                line += f"{v:>12.2f}"
        print(line)
    # how much the API actually moved the loose numbers, per checkpoint
    if any(r.get("config", {}).get("judge_api") for r in reports.values()):
        print("-" * 78)
        print(f"  {'API 仲裁':<18}" + "".join(f"{x:>12}" for x in order))
        for key, label in (("judge_arbitrated", "仲裁样本数"),
                           ("judge_flipped", "改判为正确")):
            line = f"  {label:<18}"
            for name in order:
                v = reports[name].get("config", {}).get(key)
                line += f"{v:>12}" if v is not None else f"{'-':>12}"
            print(line)
    if warns:
        print("-" * 78)
        print("  ! 对比条件警告:")
        for w in warns:
            print(f"    - {w}")
    print("=" * 78)
    print(f"  注: {n} 题时 1 个百分点 ≈ {n/100:.1f} 题；"
          "RL 增益应看 rl − sft，而非 rl − base")


def _self_test() -> None:
    def rep(p, greedy, n=458, **cfg):
        base = {"pool": "x", "samples": 8, "temperature": 0.7, "seed": 0,
                "top_p": 0.95, "max_turns": 6, "max_model_len": 4096,
                "max_new": 1024, "greedy": True, "judge_api": False}
        base.update(cfg)
        return {"overall": {"problems": n, "greedy_strict": greedy,
                            "greedy_loose": greedy + 0.08,
                            "pass@1_strict": greedy, "pass@1_loose": greedy + 0.08,
                            "pass@8_strict": greedy + 0.2,
                            "pass@8_loose": greedy + 0.28,
                            "boxed_rate": 0.9, "avg_turns": 2.0,
                            "tool_ok_per_traj": 0.8,
                            "cut_context": 0.01, "cut_budget": 0.02},
                "config": base}

    rs = {"base": rep(0.40, 0.40), "sft": rep(0.55, 0.55), "rl": rep(0.59, 0.59)}
    assert comparable(rs) == [], comparable(rs)
    rows = build_table(rs, ["base", "sft", "rl"])
    assert rows[0]["key"] == "greedy_strict"
    # RL gain is measured against SFT: 59 - 55 = 4 points, not 59 - 40
    d = delta_row(rows[0], "sft", "rl", 458).strip()
    assert d.startswith("+4.00"), d
    d2 = delta_row(rows[0], "base", "rl", 458).strip()
    assert d2.startswith("+19.00"), d2
    assert "~" not in d2, d2          # 19 points on 458 problems is a real gap
    # a one-point gap on 458 problems is inside noise and must be flagged
    tiny = {"a": rep(0.50, 0.50), "b": rep(0.51, 0.51)}
    t = build_table(tiny, ["a", "b"])[0]
    assert "~" in delta_row(t, "a", "b", 458), delta_row(t, "a", "b", 458)
    # a large gap is not flagged
    big = {"a": rep(0.30, 0.30), "b": rep(0.60, 0.60)}
    tb = build_table(big, ["a", "b"])[0]
    assert "~" not in delta_row(tb, "a", "b", 458)
    # mismatched settings are surfaced, not silently averaged
    bad = {"a": rep(0.5, 0.5), "b": rep(0.5, 0.5, seed=7)}
    assert any("seed" in w for w in comparable(bad))
    diff_n = {"a": rep(0.5, 0.5, n=458), "b": rep(0.5, 0.5, n=200)}
    assert any("题数" in w for w in comparable(diff_n))
    assert abs(se(0.5, 400) - 0.025) < 1e-9

    # a silent fallback to rules must be caught, or the loose numbers differ
    ok_j = {"a": rep(0.5, 0.5, judge_api=True, judge_used=True, judge_model="m"),
            "b": rep(0.5, 0.5, judge_api=True, judge_used=True, judge_model="m")}
    assert comparable(ok_j) == [], comparable(ok_j)
    fell_back = {"a": rep(0.5, 0.5, judge_api=True, judge_used=True, judge_model="m"),
                 "b": rep(0.5, 0.5, judge_api=True, judge_used=False, judge_model="m")}
    w = comparable(fell_back)
    assert any("口径不一致" in x for x in w), w
    assert any("实际未执行" in x for x in w), w
    # a different arbiter model is also a different standard
    diff_m = {"a": rep(0.5, 0.5, judge_api=True, judge_used=True, judge_model="m1"),
              "b": rep(0.5, 0.5, judge_api=True, judge_used=True, judge_model="m2")}
    assert any("judge_model" in x for x in comparable(diff_m))

    print("  compare self-test OK")
    print("    RL delta measured against SFT; noise flag; config mismatch surfaced")
    print("    judge fallback and arbiter-model mismatch are caught")
    print_table(rs, ["base", "sft", "rl"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="*",
                    help="name=path pairs, in order: base=... sft=... rl=...")
    ap.add_argument("--out", default=None, help="also write the table as JSON")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return 0
    if not a.reports:
        raise SystemExit("give name=path pairs, e.g. base=runs/eval_base "
                         "sft=runs/eval_sft rl=runs/eval")
    reports, order = {}, []
    for item in a.reports:
        if "=" not in item:
            raise SystemExit(f"expected name=path, got {item!r}")
        name, path = item.split("=", 1)
        reports[name] = load_report(path)
        order.append(name)
    print_table(reports, order)
    if a.out:
        payload = {"order": order,
                   "rows": build_table(reports, order),
                   "warnings": comparable(reports),
                   "overall": {k: v["overall"] for k, v in reports.items()}}
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  对比数据 -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())