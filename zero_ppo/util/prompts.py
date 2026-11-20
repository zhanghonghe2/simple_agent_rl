#!/usr/bin/env python3
"""Render prompts from the data, never from a prompt baked into the code.

The SFT corpus is ShareGPT-shaped: a top-level `system` string plus `conversations`
alternating human / gpt, where a tool result is just another human turn whose value
is `<output>...</output>`. RL has to reproduce that rendering exactly, or the policy
sees a conversation shape it was never trained on.

So this module holds NO prompt text. It only maps a record to messages and lets the
tokenizer's chat template do the rendering - the same call the SFT trainer makes.
The accepted record shapes are:

    {"conversations": [{"from": "human"|"gpt", "value": str}, ...], "system": str}
    {"question": str, "gold": str, "system": str}          # question already folded
    {"raw_question": str, "question": str, "system": str}  # bare text preferred

`prompt` is accepted as an alias of the first user turn. Nothing here invents a
protocol: if the data has no system text, no system turn is emitted.

    python3 -m zero_ppo.util.prompts --self-test
"""
from __future__ import annotations

_ROLES = {"human": "user", "gpt": "assistant", "user": "user",
          "assistant": "assistant", "system": "system"}


def messages_of(record: dict) -> list[dict]:
    """The record as chat messages, taking every string from the data itself.

    A ShareGPT `conversations` list is used verbatim (role names mapped), so an RL
    rollout starts from exactly the turns SFT trained on. Otherwise the first user
    turn is the record's own question text - `raw_question` when present, since
    `question` may already carry a folded instruction that must not be stated twice.
    """
    msgs = []
    system = record.get("system")
    if system:
        msgs.append({"role": "system", "content": system})

    convo = record.get("conversations")
    if convo:
        for turn in convo:
            role = _ROLES.get(turn.get("from", ""), None)
            if role is None:
                raise ValueError(f"unknown conversation role {turn.get('from')!r}")
            if role == "system":
                # a system turn inside the list wins over the top-level field
                if msgs and msgs[0]["role"] == "system":
                    msgs[0] = {"role": "system", "content": turn.get("value", "")}
                else:
                    msgs.insert(0, {"role": "system", "content": turn.get("value", "")})
                continue
            msgs.append({"role": role, "content": turn.get("value", "")})
        return msgs

    user = (record.get("raw_question") or record.get("prompt")
            or record.get("question", ""))
    msgs.append({"role": "user", "content": user})
    return msgs


def question_of(record: dict) -> str:
    """The first user turn's text. Kept for callers that only need the problem."""
    for m in messages_of(record):
        if m["role"] == "user":
            return m["content"]
    return ""


def render(tok, messages: list[dict], *, add_generation_prompt: bool = True) -> str:
    """One rendering path: the tokenizer's chat template, as SFT uses it.

    Without a chat template the messages are joined plainly, which is the only
    sensible fallback - no marker is invented here.
    """
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                       tokenize=False)
    return "\n\n".join(m["content"] for m in messages)


def opening_messages(record: dict) -> list[dict]:
    """The messages a rollout starts from: system (if any) + the first user turn.

    A ShareGPT record carries the whole demonstration, but RL must generate the
    assistant turns itself - feeding them in would leak the answer and leave no
    room for a policy to act.
    """
    msgs = []
    for m in messages_of(record):
        msgs.append(m)
        if m["role"] == "user":
            break
    return msgs


def build_prompt(tok, record: dict, use_chat: bool = True) -> str:
    """The opening prompt for one problem, rendered from the record's own text."""
    msgs = opening_messages(record)
    if not use_chat:
        return "\n\n".join(m["content"] for m in msgs)
    return render(tok, msgs)


class ToolFormat:
    """Extend a conversation the way the SFT data does: close gpt, add human, open gpt.

    Every marker comes from `apply_chat_template`, so the RL trajectory is a prefix
    of the same rendering SFT produced. Hand-written markers drifted from the
    template (an extra blank line after the assistant header) and put the policy
    off-distribution without raising anything.
    """

    def __init__(self, tok):
        self.tok = tok
        self.chat = bool(getattr(tok, "chat_template", None))

    def prompt(self, record) -> str:
        """Accepts a record, or a bare question string for callers that have one."""
        if isinstance(record, str):
            record = {"question": record}
        return build_prompt(self.tok, record, use_chat=self.chat)

    def extend(self, messages: list[dict], assistant_text: str,
               tool_message: str) -> tuple[list[dict], str]:
        """Append one gpt turn plus the tool's human turn.

        Returns the new message list and the exact text to append to the prompt, so
        the token ids and the string the engine sees can never diverge.
        """
        msgs = messages + [{"role": "assistant", "content": assistant_text},
                           {"role": "user", "content": tool_message}]
        if not self.chat:
            return msgs, "\n\n" + assistant_text + "\n\n" + tool_message + "\n\n"
        before = render(self.tok, messages, add_generation_prompt=True)
        after = render(self.tok, msgs, add_generation_prompt=True)
        if not after.startswith(before):
            raise RuntimeError("chat template is not prefix-stable across turns")
        return msgs, after[len(before):]


def _self_test() -> None:
    class _Tok:
        chat_template = "qwen-like"

        def apply_chat_template(self, messages, add_generation_prompt=True,
                                tokenize=False):
            s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                        for m in messages)
            return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    class _Plain:
        chat_template = None

    SYS = "protocol from the data"
    sharegpt = {"system": SYS, "conversations": [
        {"from": "human", "value": "problem text"},
        {"from": "gpt", "value": "```python\nprint(1)\n```"},
        {"from": "human", "value": "<output>1</output>"}]}
    m = messages_of(sharegpt)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"], m
    assert m[0]["content"] == SYS and m[1]["content"] == "problem text"
    assert question_of(sharegpt) == "problem text"

    # no system in the data -> no system turn invented
    assert messages_of({"question": "q"})[0]["role"] == "user"
    assert [x["role"] for x in messages_of({"question": "q"})] == ["user"]
    # bare text preferred over a folded question; prompt is an alias
    assert question_of({"question": "folded", "raw_question": "bare"}) == "bare"
    assert question_of({"prompt": "via prompt"}) == "via prompt"

    tok = _Tok()
    f = ToolFormat(tok)
    p = f.prompt(sharegpt)
    assert p.startswith("<|im_start|>system\n" + SYS) and p.endswith("<|im_start|>assistant\n")
    msgs = messages_of({"question": "q", "system": SYS})
    msgs2, tail = f.extend(msgs, "answer", "<output>1</output>")
    assert f.prompt({"question": "q", "system": SYS}) + tail == render(tok, msgs2)
    assert "<|im_start|>user\n<output>1</output>" in tail
    assert [x["role"] for x in msgs2] == ["system", "user", "assistant", "user"]

    g = ToolFormat(_Plain())
    _, plain_tail = g.extend([{"role": "user", "content": "q"}], "a", "m")
    assert plain_tail == "\n\na\n\nm\n\n"
    print("  prompts self-test OK")
    print("    ShareGPT conversations + data system used verbatim, no baked prompt")
    print("    extend() tail keeps prompt text and message rendering identical")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()