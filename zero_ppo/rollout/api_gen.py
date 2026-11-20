#!/usr/bin/env python3
"""External-model trajectory generator for LUFFY-style off-policy guidance.

The dead-lock this addresses: GRPO gives a group whose members all fail exactly
zero advantage, so problems the policy never solves produce no gradient, forever.
Mixing a few externally generated trajectories into each group restores a positive
member and lets the relative advantage teach the policy something it could not
sample on its own (LUFFY, arXiv:2504.14945; this is the simple variant - the traces
enter the group unchanged and the importance ratio is left as-is, i.e. the update
on those rows is advantage-weighted imitation rather than shaped importance
sampling).

Design constraints that keep the traces on the training distribution:

  * the API sees the EXACT prompt string the policy engine sees (system + user,
    chat-rendered), not a rephrased problem;
  * tool calls are executed by the SAME local sandbox and fed back in the SAME
    <output>...</output> user turn, through the same multiturn loop - the API never
    invents execution results;
  * the API is asked to answer in the trained format (```python blocks, \\boxed{}).

Only the two answer-free strings of the conversation are sent out: the problem and
the model's own prior turns plus real tool outputs. Gold answers are NEVER sent.

    python3 -m zero_ppo.rollout.api_gen --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# The rendered prompt ends with the assistant header; split there so the API gets
# a proper messages list instead of one blob of template markers.
_SPLIT = re.compile(r"<\|im_start\|>(system|user|assistant)\n")


def prompt_to_messages(convo: str) -> list[dict]:
    """Chat-rendered prompt -> messages, dropping the trailing assistant header.

    Falls back to a single user message when the text carries no chat markers
    (the no-template path), so both rendering modes work.
    """
    parts = _SPLIT.split(convo)
    if len(parts) < 3:
        return [{"role": "user", "content": convo.strip()}]
    msgs = []
    # parts = [pre, role1, body1, role2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        role, body = parts[i], parts[i + 1]
        body = body.replace("<|im_end|>", "").strip()
        if role == "assistant" and not body:
            continue                      # the dangling generation header
        msgs.append({"role": role, "content": body})
    return msgs or [{"role": "user", "content": convo.strip()}]


class ApiGenerator:
    """generate(prompts) -> [{"text": ...}] via an OpenAI-compatible chat API.

    Text only, no token ids: the rollout loop tokenizes API text with the training
    tokenizer, which is the correct thing for an off-policy trace anyway (the
    external model's own tokenization is not ours).
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key_env: str = "DEEPSEEK_API_KEY", temperature: float = 0.7,
                 max_tokens: int = 1024, timeout: float = 120.0, retries: int = 2,
                 workers: int = 8):
        self.model = model or os.environ.get("LUFFY_MODEL", "deepseek-chat")
        self.base_url = (base_url or os.environ.get("LUFFY_BASE_URL")
                         or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.deepseek.com/v1").rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.workers = max(1, int(workers))
        self.calls = 0
        self.errors = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _one(self, convo: str) -> str:
        import time
        import requests
        msgs = prompt_to_messages(convo)
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model, "temperature": self.temperature,
                          "max_tokens": self.max_tokens, "messages": msgs},
                    timeout=self.timeout)
                r.raise_for_status()
                self.calls += 1
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:
                last = e
                if attempt < self.retries:
                    time.sleep(min(2.0 ** attempt, 4.0))
        self.errors += 1
        if self.errors == 1:
            print(f"  ! api generator failed: {type(last).__name__}: {last}",
                  flush=True)
        return ""                          # empty -> the row ends, reward 0

    def __call__(self, prompts: list[str]) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor
        if not self.available:
            raise RuntimeError("LUFFY api generator needs the API key "
                               "(set the env var named by --luffy-key-env)")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            texts = list(pool.map(self._one, prompts))
        return [{"text": t} for t in texts]


def _self_test() -> None:
    # --- prompt splitting ---------------------------------------------------
    convo = ("<|im_start|>system\nbe helpful<|im_end|>\n"
             "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
             "<|im_start|>assistant\n")
    m = prompt_to_messages(convo)
    assert m == [{"role": "system", "content": "be helpful"},
                 {"role": "user", "content": "What is 2+2?"}], m
    # multi-turn: prior assistant turn + tool output stay in order
    convo2 = (convo.replace("<|im_start|>assistant\n", "") +
              "<|im_start|>assistant\n```python\nprint(4)\n```<|im_end|>\n"
              "<|im_start|>user\n<output>4</output><|im_end|>\n"
              "<|im_start|>assistant\n")
    m2 = prompt_to_messages(convo2)
    assert [x["role"] for x in m2] == ["system", "user", "assistant", "user"], m2
    assert m2[-1]["content"] == "<output>4</output>"
    # no markers -> single user message
    assert prompt_to_messages("plain question") == [
        {"role": "user", "content": "plain question"}]

    # --- generator with a stubbed transport ---------------------------------
    os.environ["LUFFY_TEST_KEY"] = "x"
    g = ApiGenerator(api_key_env="LUFFY_TEST_KEY")   # key is read at construction
    seen = []

    def fake(convo):
        seen.append(convo)
        return "reasoning \\boxed{4}"

    g._one = fake
    out = g(["p1", "p2"])
    assert out == [{"text": "reasoning \\boxed{4}"}] * 2
    assert seen == ["p1", "p2"]
    del os.environ["LUFFY_TEST_KEY"]

    # missing key must raise, not silently return empty traces
    g2 = ApiGenerator(api_key_env="LUFFY_TEST_MISSING_KEY")
    try:
        g2(["p"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing key must raise")

    # --- integration: api rows flow through the real multiturn loop ---------
    from zero_ppo.rollout.multiturn import rollout_batch
    from zero_ppo.util.prompts import ToolFormat

    class _Tok:
        chat_template = "yes"

        def apply_chat_template(self, messages, add_generation_prompt=True,
                                tokenize=False):
            s = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
            return s + ("<|assistant|>" if add_generation_prompt else "")

    tok = _Tok()
    fmt = ToolFormat(tok)
    enc = lambda t: [ord(c) % 97 + 3 for c in t]

    def pol_gen(prompts):
        # policy: one code turn then answer, per prompt
        return [{"text": "```python\nprint(2+2)\n```", "tokens": [1, 2]}
                if "<output>" not in p else
                {"text": "The answer is \\boxed{4}", "tokens": [3]}
                for p in prompts]

    api_calls = []

    def api_gen(prompts):
        api_calls.append(len(prompts))
        return [{"text": "```python\nprint(4)\n```"} if "<output>" not in p
                else {"text": "So \\boxed{4}"} for p in prompts]

    def run_code(src):
        return ("4", None)

    probs = [{"question": "2+2?", "gold": "4"}] * 4
    rows = rollout_batch(probs, pol_gen, run_code,
                         lambda pr: fmt.prompt(pr), enc, fmt=fmt,
                         generate_api=api_gen, api_mask=[False, True, False, True],
                         max_turns=3)
    assert [r["source"] for r in rows] == ["policy", "api", "policy", "api"]
    assert api_calls == [2, 2], api_calls       # two api rows, two turns
    for r in rows:
        assert len(r["texts"]) == 2, r["texts"]
        assert "\\boxed{4}" in r["texts"][-1]
        assert r["seg_ok"][0] is True           # sandbox ran for BOTH sources
    # api rows got their ids from encode(text), policy rows from engine tokens
    assert rows[1]["ids"] != rows[0]["ids"]

    # no mask -> identical to the old behaviour, no api calls
    rows2 = rollout_batch(probs, pol_gen, run_code, lambda pr: fmt.prompt(pr),
                          enc, fmt=fmt, max_turns=3)
    assert all(r["source"] == "policy" for r in rows2)

    print("  api_gen self-test OK")
    print("    chat prompt -> messages, gold never sent, missing key raises")
    print("    api rows share the sandbox/template loop with policy rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()