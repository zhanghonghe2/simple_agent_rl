#!/usr/bin/env python3
"""Multi-turn rollout: generate, execute, feed the output back, generate again.

One engine call per turn for the whole wavefront. Per-problem calls were the
scheduling bottleneck: thirty-two rows cost sixty-four to a hundred and
twenty-eight requests per step, and the step time grew without any number looking
wrong.

Two shapes are kept deliberately:

  * a trajectory always ends on an assistant turn. Appending a tool output after
    the model has already answered leaves the sequence ending on a user turn, and
    a reward placed at the end then lands on a token the model never produced.
  * a failed execution is reported back and the model may try again. Refusing to
    continue turns one syntax slip into a zero, and we measured that as roughly a
    tenth of all problems.

    python3 -m zero_ppo.rollout.multiturn --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
BOX = re.compile(r"\\boxed\{")

#: Hard cap on assistant turns per trajectory. A model that keeps calling the tool
#: without ever boxing an answer would otherwise loop until the token budget ran
#: out, burning a rollout slot and producing a trajectory with no scorable end.
#: Training and evaluation share this number so both see the same task.
MAX_TURNS = 6


def rollout_batch(problems, generate, run_code, prompt_of, encode, *,
                  max_turns: int = MAX_TURNS, max_tokens: int = 2400,
                  chunk: int = 8, fmt=None, generate_api=None, api_mask=None):
    """-> one row per problem: ids, spans, seg_end, seg_kinds, seg_ok, texts.

    `api_mask` marks the rows an EXTERNAL model generates (LUFFY-style off-policy
    guidance): those rows use `generate_api` instead of the policy engine, but go
    through the SAME tool loop, sandbox and template, so their trajectories sit on
    the training distribution and mix into the group's relative advantage. Rows
    are tagged `source` = "api" | "policy". With no mask everything is on-policy
    and behaviour is unchanged.
    """
    rows = []
    for k, pr in enumerate(problems):
        prefix = prompt_of(pr)
        is_api = bool(api_mask is not None and api_mask[k])
        row = {"ids": list(encode(prefix)), "spans": [], "seg_end": [],
               "seg_kinds": [], "seg_ok": [], "seg_outs": [], "texts": [],
               "convo": prefix, "source": "api" if is_api else "policy"}
        if fmt is not None:
            # the message list is the source of truth; the template renders it the
            # same way the SFT trainer did, so RL stays on the trained distribution
            from zero_ppo.util.prompts import opening_messages
            row["msgs"] = opening_messages(pr)
        rows.append(row)
    active = list(range(len(rows)))
    calls = 0

    def _gen(idxs, fn):
        nonlocal calls
        calls += 1
        return fn([rows[i]["convo"] for i in idxs])

    for _turn in range(max_turns):
        if not active:
            break
        nxt = []
        # split the wavefront by generator, then chunk each part separately
        pol = [i for i in active if rows[i]["source"] == "policy"]
        api = [i for i in active if rows[i]["source"] == "api"]
        batches = []
        for s in range(0, len(pol), chunk):
            batches.append((pol[s:s + chunk], generate))
        if api:
            if generate_api is None:
                raise ValueError("api_mask given but no generate_api callable")
            for s in range(0, len(api), chunk):
                batches.append((api[s:s + chunk], generate_api))
        for part, fn in batches:
            outs = _gen(part, fn)
            for i, g in zip(part, outs):
                r = rows[i]
                toks = list(g.get("tokens") or [])
                if not toks:
                    # the API generator returns text only: tokenize it the same
                    # way the prompt is tokenized so ids stay consistent
                    toks = list(encode(g.get("text", "")))
                if not toks or len(r["ids"]) >= max_tokens:
                    continue                       # row is done, possibly truncated
                start = len(r["ids"])
                r["ids"].extend(toks)
                r["spans"].append((start, len(r["ids"]) - 1))
                r["seg_end"].append(len(r["ids"]) - 1)
                text = g.get("text", "")
                r["texts"].append(text)
                blocks = CODE.findall(text)
                if not blocks:
                    r["seg_kinds"].append("nl")
                    r["seg_ok"].append(None)
                    r["seg_outs"].append(None)
                    continue
                out, err = run_code(blocks[-1].strip())
                ok = out is not None
                r["seg_kinds"].append("code")
                r["seg_ok"].append(ok)
                # keep the execution result so a process judge can check it
                r["seg_outs"].append(out if ok else
                                     "ERROR: " + str(err).splitlines()[0][:120])
                if BOX.search(text):
                    continue                       # answered; do not append a tool turn
                msg = ("<output>" + (out if ok else
                       "ERROR: " + str(err).splitlines()[0][:80]) + "</output>")
                # One place decides how a turn is closed and a tool reply opens:
                # with a chat template the assistant turn needs its end marker and
                # the reply arrives as a user turn, or the model reads the output as
                # a continuation of its own answer.
                if fmt is not None:
                    r["msgs"], tail = fmt.extend(r["msgs"], text, msg)
                    r["convo"] = r["convo"] + tail
                    # Keep the engine's own generated tokens (re-encoding the text
                    # could split it differently) and append only what the template
                    # adds after them: the turn end marker and the tool's user turn.
                    added = tail[len(text):] if tail.startswith(text) else tail
                    r["ids"].extend(encode(added))
                else:
                    r["convo"] = r["convo"] + "\n\n" + text + "\n\n" + msg
                    r["ids"].extend(encode("\n\n" + msg))
                nxt.append(i)
        active = nxt
    for r in rows:
        r.pop("convo", None)
        r.pop("msgs", None)
    rollout_batch.last_calls = calls
    return rows


# --------------------------------------------------------------------- self test
def _self_test() -> None:
    FENCE = "```python\nprint(6*7)\n```"
    BOXED = "\\" + "boxed{42}"
    script = {
        "p0": [{"text": "need it\n" + FENCE, "tokens": [1, 2, 3]},
               {"text": "so " + BOXED, "tokens": [4, 5]}],
        "p1": [{"text": FENCE + " and " + BOXED, "tokens": [6, 7]}],   # answers at once
        "p2": [{"text": "```python\nprint(1/0)\n```", "tokens": [8]},
               {"text": FENCE, "tokens": [9, 10]},
               {"text": BOXED, "tokens": [11]}],
    }

    state = {"calls": 0}

    def generate(prompts):
        state["calls"] += 1
        out = []
        for p in prompts:
            key = next(k for k in script if k in p)
            turn = sum(1 for line in p.splitlines() if "assistant" in line
                       or "print" in line)
            seq = script[key]
            out.append(seq[min(turn, len(seq) - 1)])
        return out

    from zero_ppo.rollout.tools import fake_runner
    fake = fake_runner({"1/0": (None, "ZeroDivisionError: division by zero"),
                        "print": ("42", "")})

    rows = rollout_batch([{"q": "p0"}, {"q": "p1"}, {"q": "p2"}], generate, fake,
                         lambda pr: pr["q"], lambda s: [ord(c) % 97 + 1 for c in s],
                         max_turns=4, chunk=8)

    # one engine call per turn, not per problem
    assert state["calls"] == 3, state["calls"]

    # p0: code then answer -> two segments, ends on the assistant turn
    assert rows[0]["seg_kinds"] == ["code", "nl"], rows[0]["seg_kinds"]
    assert rows[0]["seg_ok"] == [True, None]
    assert rows[0]["spans"][-1][1] == len(rows[0]["ids"]) - 1

    # p1: answered in the same turn as the code -> ONE segment, no tool turn
    assert rows[1]["seg_kinds"] == ["code"], rows[1]["seg_kinds"]
    assert rows[1]["spans"][-1][1] == len(rows[1]["ids"]) - 1
    # the program is still executed (its success is what a process reward scores),
    # but no tool OUTPUT is appended, so the trajectory ends on the assistant
    assert any("print(6*7)" in c for c in fake.calls)
    assert len(rows[1]["spans"]) == 1, rows[1]["spans"]

    # p2: fails, retries, answers -> the failure is recorded and recovery happened
    assert rows[2]["seg_ok"][0] is False and rows[2]["seg_ok"][1] is True
    assert rows[2]["seg_kinds"] == ["code", "code", "nl"]

    print("  rollout self-test OK")
    print(f"    3 problems, one engine call per turn: {state['calls']} calls")
    print("    answered-in-the-same-turn ends on the assistant, no tool turn")
    print("    a failed execution is reported and the retry is recorded")


def _self_test_alignment() -> None:
    """The trained ids must spell out exactly the text the engine was sent.

    A chat template inserts turn markers between the generated text and the tool
    reply. If those markers reach `convo` but not `ids`, every later position
    shifts and the reward lands on tokens the model never produced - silently.
    A character-level encoder makes the two directly comparable.
    """
    from zero_ppo.util.prompts import ToolFormat
    from zero_ppo.rollout.tools import fake_runner

    class _Chat:
        chat_template = "yes"
        eos_token = "<|im_end|>"

        def apply_chat_template(self, messages, add_generation_prompt=True,
                                tokenize=False):
            s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                        for m in messages)
            return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    fmt = ToolFormat(_Chat())
    encode = lambda s: [ord(c) for c in s]        # 1 char = 1 "token"
    FENCE = "```python\nprint(1)\n```"
    seen = []

    def generate(prompts):
        seen.extend(prompts)
        # answer on the second turn so the trajectory ends on an assistant turn
        text = FENCE if len(seen) == 1 else "\\" + "boxed{1}"
        return [{"text": text, "tokens": encode(text)} for _ in prompts]

    # a ShareGPT-shaped record: the system text and the question come from the data
    rec = {"system": "protocol from data", "question": "what is 1?"}
    rows = rollout_batch([rec], generate, fake_runner({"print": ("1", "")}),
                         fmt.prompt, encode, max_turns=3, chunk=8, fmt=fmt)
    r = rows[0]
    # rebuild the full text: the last prompt sent + the final generation
    full = seen[-1] + "\\" + "boxed{1}"
    assert len(r["ids"]) == len(full), (len(r["ids"]), len(full))
    assert "".join(chr(t) for t in r["ids"]) == full, "ids diverge from the prompt text"
    # the tool reply really is a user turn, and the assistant turn was closed
    assert "<|im_start|>user\n<output>1</output>" in full
    assert FENCE + "<|im_end|>" in full
    # the data's own system text is present, and no prompt was invented
    assert full.startswith("<|im_start|>system\nprotocol from data")
    # segment ends still point at the last generated token of each turn
    assert r["seg_end"][0] == full.index(FENCE) + len(FENCE) - 1
    print("  rollout alignment self-test OK")
    print("    ids spell out the exact prompt text, data system + chat markers included")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        _self_test_alignment()
    else:
        ap.print_help()
