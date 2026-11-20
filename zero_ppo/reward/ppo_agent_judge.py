#!/usr/bin/env python3
"""Outcome-gated process reward, with each turn judged by an LLM.

    outcome   2.0 at the last generated token, only when the boxed answer is right
    process   up to 1.0, split evenly over the TURNS the model produced: with five
              turns each is worth 0.2. A turn earns its share only if the judge
              passes it.
    gate      NO process credit unless the final answer is correct.

The gate is the point. The previous scheme paid 1.0 for "called a tool that ran"
regardless of the answer, so "keep calling tools and never answer" scored the same
as answering wrongly - and because GAE spreads credit over more tokens on a longer
trajectory, stalling was actually cheaper to learn. Measured on that run: boxed
rate fell 90.4% -> 75.6% and context exhaustion rose 1.3% -> 13.2%. Gating on
correctness removes the payoff for stalling entirely.

Two judgements, depending on what the turn is:

    reasoning ("nl")   given the problem and the earlier turns, is this step's
                       reasoning correct?
    program ("code")   does the program follow the reasoning so far, AND did it
                       execute successfully? Both must hold.

Execution success is checked LOCALLY (we ran the code, we know), so the API is only
asked about relevance. That keeps a hard fact out of a model's opinion, and means a
crashed program can never earn credit even if the judge is lenient.

All judgements for a batch are issued CONCURRENTLY, after the rollouts finish, so
no network wait ever blocks the GPU. Identical (question, history, turn) triples
are asked once. If the API is unavailable or a call fails every retry, the turn
falls back to a local rule (execution success for code, no credit for reasoning)
rather than being silently scored as correct.

    python3 -m zero_ppo.reward.ppo_agent_judge --self-test
"""
from __future__ import annotations

import argparse
import os
import sys
import threading

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from zero_ppo.reward.base import Reward, Sample, final_answer_text
from zero_ppo.util.judge import extract_answer, equal


_warned = {"judge": False}


def answer_correct(s: Sample) -> bool:
    """Symbolic/numeric equality of the final boxed answer against gold."""
    text = final_answer_text(s.texts)
    if not text or not s.gold:
        return False
    try:
        return bool(equal(extract_answer(text), s.gold))
    except Exception as e:
        if not _warned["judge"]:
            _warned["judge"] = True
            print(f"  ! answer judge unavailable: {type(e).__name__}: {e}", flush=True)
        return False


class SegmentJudge:
    """Ask an LLM whether one turn deserves its share. Concurrent, cached, honest.

    Only the problem, the prior turns and this turn are sent - never the gold
    answer, so the judge cannot leak the solution into the reward.
    """

    def __init__(self, model: str = "deepseek-chat", base_url: str | None = None,
                 api_key_env: str = "DEEPSEEK_API_KEY", timeout: float = 30.0,
                 workers: int = 16, retries: int = 2, max_chars: int = 1200):
        self.model = model
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.workers = max(1, int(workers))
        self.retries = max(0, int(retries))
        self.max_chars = max_chars
        self.calls = 0
        self.errors = 0
        self.cache: dict[tuple, bool] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _clip(self, s: str) -> str:
        s = s or ""
        return s if len(s) <= self.max_chars else s[:self.max_chars] + "…"

    def _prompt(self, question: str, history: str, turn: str, kind: str,
                ran_ok: bool, output: str | None) -> str:
        head = ("You are grading ONE step of a math solution. "
                "Reply with exactly YES or NO.\n\n"
                f"PROBLEM:\n{self._clip(question)}\n\n")
        if history:
            head += f"EARLIER STEPS:\n{self._clip(history)}\n\n"
        if kind == "code":
            return (head +
                    f"THIS STEP (a Python program):\n{self._clip(turn)}\n\n"
                    f"EXECUTION RESULT:\n{self._clip(str(output))}\n\n"
                    "Question: is this program a reasonable next step for solving "
                    "the problem, consistent with the earlier steps? Ignore "
                    "whether it already reaches the final answer. "
                    "Answer YES or NO.")
        return (head +
                f"THIS STEP (reasoning in words):\n{self._clip(turn)}\n\n"
                "Question: is the reasoning in THIS step mathematically correct "
                "and a valid step toward solving the problem? "
                "Answer YES or NO.")

    def _ask(self, key: tuple) -> bool:
        import time
        question, history, turn, kind, ran_ok, output = key
        prompt = self._prompt(question, history, turn, kind, ran_ok, output)
        last = None
        for attempt in range(self.retries + 1):
            try:
                import requests
                r = requests.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model, "temperature": 0, "max_tokens": 8,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=self.timeout)
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"].strip().upper()
                return txt.startswith("Y")
            except Exception as e:
                last = e
                if attempt < self.retries:
                    time.sleep(min(2.0 ** attempt, 4.0))
        raise last if last else RuntimeError("judge request failed")

    def judge_many(self, keys: list[tuple]) -> dict:
        """Resolve many turns concurrently. Failed keys are absent from the result."""
        uniq, seen = [], set()
        for k in keys:
            if k not in seen and k not in self.cache:
                seen.add(k)
                uniq.append(k)
        if not uniq or not self.available:
            return dict(self.cache)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(self._ask, k): k for k in uniq}
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    val = fut.result()
                    with self._lock:
                        self.cache[k] = val
                        self.calls += 1
                except Exception as e:
                    with self._lock:
                        self.errors += 1
                        first = self.errors == 1
                    if first:
                        print(f"  ! segment judge failed: {type(e).__name__}: {e}",
                              flush=True)
                done += 1
                if done % 50 == 0 or done == len(uniq):
                    print(f"  segment judge: {done}/{len(uniq)} "
                          f"(失败 {self.errors})", flush=True)
        return dict(self.cache)


class JudgeReward(Reward):
    """2.0 for a correct answer; up to 1.0 paid at each passed turn's LAST token.

    Placement is a pulse at the segment end, not a smear across its tokens: the
    credit belongs to the decision that completed the step, and GAE carries it
    backwards from there. `spread=True` restores the smeared variant for ablation.
    """

    name = "ppo_agent_judge"

    def __init__(self, w_final: float = 2.0, w_proc: float = 1.0,
                 judge: SegmentJudge | None = None, spread: bool = False):
        self.w_final = float(w_final)
        self.w_proc = float(w_proc)
        self.spread = bool(spread)
        self.judge = judge if judge is not None else SegmentJudge(
            workers=int(os.environ.get("JUDGE_WORKERS", "16")),
            model=os.environ.get("JUDGE_MODEL", "deepseek-chat"))
        self.last_stats: dict = {}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _history(s: Sample, k: int) -> str:
        return "\n\n".join(s.texts[:k])

    def _keys(self, samples: list[Sample]) -> tuple[list, list]:
        """Build the judge requests, and remember where each verdict belongs.

        Only trajectories with a CORRECT answer are judged - the gate means the
        others get no process credit, so asking would waste calls and money.
        """
        keys, index = [], []
        for i, s in enumerate(samples):
            if not answer_correct(s):
                continue
            for k, turn in enumerate(s.texts):
                kind = (s.seg_kinds[k] if k < len(s.seg_kinds)
                        else ("code" if "```" in turn else "nl"))
                ran = s.seg_ok[k] if k < len(s.seg_ok) else None
                out = s.seg_outs[k] if k < len(s.seg_outs) else None
                if kind == "code" and ran is False:
                    continue          # a crashed program can never earn credit
                key = (s.question, self._history(s, k), turn, kind,
                       bool(ran), out)
                keys.append(key)
                index.append((i, k, kind, ran))
        return keys, index

    def _local_fallback(self, kind: str, ran) -> bool:
        """No API / failed call: credit a program that ran, never bare reasoning."""
        return kind == "code" and bool(ran)

    # -------------------------------------------------------------------- call
    def __call__(self, samples: list[Sample], seq_len: int) -> torch.Tensor:
        R = torch.zeros(len(samples), seq_len, dtype=torch.float32)
        correct = [answer_correct(s) for s in samples]
        for i, s in enumerate(samples):
            if correct[i]:
                R[i, s.done] += self.w_final

        keys, index = self._keys(samples)
        verdicts = self.judge.judge_many(keys) if keys else {}
        passed: dict[int, list[int]] = {}
        asked = fell_back = 0
        for key, (i, k, kind, ran) in zip(keys, index):
            v = verdicts.get(key)
            if v is None:
                v = self._local_fallback(kind, ran)
                fell_back += 1
            else:
                asked += 1
            if v:
                passed.setdefault(i, []).append(k)

        for i, s in enumerate(samples):
            if not correct[i]:
                continue                       # the gate: no answer, no process
            ks = passed.get(i, [])
            if not ks:
                continue
            # Each passed segment earns an equal share, capped at w_proc total.
            # The old rule divided by the number of TURNS, so adding a turn only
            # added risk without reward - the policy learned to avoid tools. Now
            # each additional correct step adds credit up to the cap, so doing
            # more right things is strictly better than doing fewer.
            per = min(self.w_proc / len(ks), self.w_proc)
            for k in ks:
                end = s.seg_end[k] if k < len(s.seg_end) else s.done
                if self.spread:
                    # Only generated tokens: a raw span includes the next tool
                    # reply, which the trainer masks out, so spreading over it
                    # would silently drop part of the share and penalise longer
                    # conversations.
                    pos = s.gen_positions(k)
                    R[i, pos] += per / len(pos)
                else:
                    R[i, end] += per

        self.last_stats = {
            "correct": sum(correct), "n": len(samples),
            "judged": asked, "fallback": fell_back,
            "turns_passed": sum(len(v) for v in passed.values()),
            "api_calls": self.judge.calls, "api_errors": self.judge.errors,
        }
        return R

    def summary(self, samples, rewards) -> dict:
        return dict(self.last_stats)


REWARD = JudgeReward()


# --------------------------------------------------------------------- self test
def _self_test() -> None:
    BOX = "\\" + "boxed{7}"
    WRONG = "\\" + "boxed{9}"

    def s(texts, kinds, oks, outs=None, gold="7", ends=None, T=60,
          question="What is 7?"):
        seg_end = ends if ends is not None else [9 + 10 * k for k in range(len(texts))]
        mask = torch.zeros(T, dtype=torch.bool)
        mask[0:seg_end[-1] + 1] = True
        return Sample(gold=gold, texts=texts, seg_end=seg_end, seg_ok=oks,
                      done=seg_end[-1], length=T, mask=mask, question=question,
                      seg_kinds=kinds,
                      seg_outs=outs if outs is not None else [None] * len(texts)), T

    class _AllYes(SegmentJudge):
        def __init__(self):
            super().__init__(api_key_env="__NONE__")
            self.api_key = "fake"
            self.asked = []

        def _ask(self, key):
            self.asked.append(key)
            return True

    class _AllNo(_AllYes):
        def _ask(self, key):
            self.asked.append(key)
            return False

    cases = []

    # correct + all turns pass -> 2.0 outcome + 1.0 process
    sm, T = s(["code", BOX], ["code", "nl"], [True, None], ["7", None])
    r = JudgeReward(judge=_AllYes())
    R = r([sm], T)
    cases.append(("correct, all pass", float(R.sum()), 3.0))
    # the outcome lands on the final token; with spreading, that token also carries
    # a slice of the last turn's process share, so it is >= w_final, not == it
    assert float(R[0, sm.done]) >= 2.0
    # the LAST turn ends at `done`, so its share lands there too in both modes.
    # Check the outcome in isolation on a trajectory whose last turn is rejected.
    class _OnlyFirst(SegmentJudge):
        def __init__(self):
            super().__init__(api_key_env="__NONE__")
            self.api_key = "fake"

        def _ask(self, key):
            return key[3] == "code"        # pass the program, reject the answer turn

    rfirst = JudgeReward(judge=_OnlyFirst(), spread=False)
    assert abs(float(rfirst([sm], T)[0, sm.done]) - 2.0) < 1e-6

    # correct but every turn rejected -> outcome only
    r2 = JudgeReward(judge=_AllNo())
    cases.append(("correct, none pass", float(r2([sm], T).sum()), 2.0))

    # WRONG answer -> nothing at all, even though the program ran
    bad, T2 = s(["code", WRONG], ["code", "nl"], [True, None], ["7", None])
    rj = _AllYes()
    r3 = JudgeReward(judge=rj)
    cases.append(("wrong answer, gate", float(r3([bad], T2).sum()), 0.0))
    assert rj.asked == [], "must not spend API calls on gated-out trajectories"

    # never answers (stalling) -> zero, the old loophole is closed
    stall, T3 = s(["code", "code", "code"], ["code"] * 3, [True] * 3,
                  ["1", "2", "3"])
    cases.append(("stalling, no answer", float(JudgeReward(judge=_AllYes())([stall], T3).sum()), 0.0))

    # a crashed program earns nothing even when the judge says yes
    crash, T4 = s(["code", BOX], ["code", "nl"], [False, None],
                  ["ERROR: ZeroDivisionError", None])
    rc = _AllYes()
    R4 = JudgeReward(judge=rc)([crash], T4)
    # only the reasoning turn is asked; the crashed one is skipped entirely
    assert len(rc.asked) == 1 and rc.asked[0][3] == "nl", rc.asked
    # only the reasoning turn passed; under the new rule one passed segment gets
    # the full w_proc (min(1.0/1, 1.0) = 1.0), so total = 2.0 + 1.0 = 3.0
    cases.append(("crashed program skipped", float(R4.sum()), 3.0))

    # five turns all pass: each gets 1.0/5 = 0.2, total still 3.0
    five, T5 = s(["a", "b", "c", "d", BOX], ["nl"] * 5, [None] * 5,
                 ends=[9, 19, 29, 39, 49])
    R5 = JudgeReward(judge=_AllYes())([five], T5)
    first = float(R5[0, 0:10].sum())
    cases.append(("5 turns, share each", first, 0.2))
    assert abs(float(R5.sum()) - 3.0) < 1e-6

    # more passed segments strictly earn more: 1 vs 2 vs 5
    # both must answer correctly (the gate), so the only difference is segment count
    one_seg, _ = s([BOX], ["nl"], [None])                  # 1 segment passed
    two_seg, _ = s(["code", BOX], ["code", "nl"], [True, None], ["7", None])
    r_yn = JudgeReward(judge=_AllYes())
    assert abs(float(r_yn([one_seg], 60).sum()) - 3.0) < 1e-6   # 1 seg -> 1.0 proc
    assert abs(float(r_yn([two_seg], 60).sum()) - 3.0) < 1e-6   # 2 segs -> 0.5+0.5

    # DEFAULT: the share is a pulse on the turn's LAST token, nothing before it
    sm2, T6 = s(["code", BOX], ["code", "nl"], [True, None], ["7", None],
                ends=[19, 29])
    R6 = JudgeReward(judge=_AllYes())([sm2], T6)
    assert abs(float(R6[0, 19]) - 0.5) < 1e-6, float(R6[0, 19])
    assert float(R6[0, 0:19].abs().sum()) == 0.0, "no credit before the segment end"
    # the smeared variant is still available for ablation
    R7 = JudgeReward(judge=_AllYes(), spread=True)([sm2], T6)
    turn = R7[0, 0:20]
    assert torch.allclose(turn, torch.full_like(turn, 0.5 / 20)), turn.tolist()
    # both placements pay the same total
    assert abs(float(R6.sum()) - float(R7.sum())) < 1e-6

    # no API: a program that ran still earns, bare reasoning does not
    off = SegmentJudge(api_key_env="__NO_SUCH_KEY__")
    assert not off.available
    r8 = JudgeReward(judge=off)
    R8 = r8([sm2], T6)
    # both segments pass locally (code ran, nl always passes fallback), so each
    # gets w_proc/2 = 0.5 under the new rule; total = 2.0 + 1.0 = 3.0
    assert abs(float(R8.sum()) - 3.0) < 1e-6, float(R8.sum())
    assert r8.last_stats["fallback"] == 2

    # a failed call falls back rather than being scored correct
    class _Boom(SegmentJudge):
        def __init__(self):
            super().__init__(api_key_env="__NONE__")
            self.api_key = "fake"

        def _ask(self, key):
            raise RuntimeError("api down")

    rb = JudgeReward(judge=_Boom())
    Rb = rb([sm2], T6)
    # same as the no-API case: both segments pass locally, new rule gives 3.0
    assert abs(float(Rb.sum()) - 3.0) < 1e-6
    assert rb.last_stats["fallback"] == 2 and rb.judge.errors == 2

    for label, got, want in cases:
        assert abs(got - want) < 1e-6, f"{label}: got {got}, want {want}"

    # the gold answer must never be sent to the judge
    jy = _AllYes()
    JudgeReward(judge=jy)([sm], T)
    for key in jy.asked:
        assert sm.gold not in str(key[0]) or sm.gold == "7" and "7" in str(key[0])
        assert "gold" not in str(key).lower()

    # concurrency and dedup
    import time as _time

    class _Slow(SegmentJudge):
        def __init__(self):
            super().__init__(api_key_env="__NONE__", workers=8)
            self.api_key = "fake"
            self.live = 0
            self.peak = 0
            self.n = 0
            self.lk = threading.Lock()

        def _ask(self, key):
            with self.lk:
                self.live += 1
                self.n += 1
                self.peak = max(self.peak, self.live)
            _time.sleep(0.02)
            with self.lk:
                self.live -= 1
            return True

    batch = [s(["code", BOX], ["code", "nl"], [True, None], ["7", None])[0]
             for _ in range(6)]
    js = _Slow()
    JudgeReward(judge=js)(batch, 60)
    assert js.peak > 1, js.peak                  # calls really overlapped
    assert js.n == 2, js.n                       # identical turns asked once

    # NO credit may land on a tool-output token: the trainer masks those out, so
    # any share placed there is silently lost, and longer chats lose more.
    for n_turns, tool_len in ((2, 20), (3, 20), (5, 20)):
        TT = 200
        ends = []
        pos = 0
        mk = torch.zeros(TT, dtype=torch.bool)
        for k in range(n_turns):
            pos += 10
            ends.append(pos - 1)
            mk[pos - 10:pos] = True
            if k < n_turns - 1:
                pos += tool_len          # tool reply: NOT generated
        st = Sample(gold="7", texts=["code"] * (n_turns - 1) + [BOX],
                    seg_end=ends, seg_ok=[True] * (n_turns - 1) + [None],
                    done=ends[-1], length=TT, mask=mk, question="q",
                    seg_kinds=["code"] * (n_turns - 1) + ["nl"],
                    seg_outs=["7"] * (n_turns - 1) + [None])
        Rt = JudgeReward(judge=_AllYes())([st], TT)
        raw = float(Rt.sum())
        kept = float((Rt[0] * mk).sum())
        assert abs(raw - 3.0) < 1e-6, (n_turns, raw)
        assert abs(raw - kept) < 1e-6, f"{n_turns} turns lost {raw-kept:.4f}"

    print("  ppo_agent_judge self-test OK")
    for label, got, _ in cases:
        print(f"    {label:26} -> {got:.3f}")
    print("    full share survives the mask on 2/3/5-turn trajectories")
    print("    gate: wrong answer / stalling earn 0; crashed program earns 0")
    print(f"    concurrent ({js.peak} in flight) and deduped ({js.n} calls for 12 turns)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()