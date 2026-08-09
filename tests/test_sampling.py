from imessage_mlx.data.sft import SFT_FORMAT
from imessage_mlx.sampling import (
    DEMO_BOUNDARY,
    DEMO_HEADER,
    _select_targets,
    _trim_completion,
    build_generation_prompt,
    build_personalized_prompt,
)


def test_build_generation_prompt_stops_before_gold_reply() -> None:
    messages = [
        {"role": "user", "content": "free later?"},
        {"role": "assistant", "content": "yeah around 7"},
        {"role": "user", "content": "cool"},
        {"role": "assistant", "content": "lmk when"},
    ]
    prompt, gold = build_generation_prompt(messages, 3)
    assert gold == "lmk when"
    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_start|>assistant\nyeah around 7<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "lmk when" not in prompt


def test_trim_completion_cuts_at_turn_end() -> None:
    assert _trim_completion("sounds good<|im_end|>\n<|im_start|>user\nnope") == "sounds good"


def test_personalized_prompt_orders_demos_style_and_live_context() -> None:
    messages = [
        {"role": "user", "content": "current question"},
        {"role": "assistant", "content": "gold reply"},
    ]

    prompt, gold = build_personalized_prompt(
        messages,
        1,
        retrieved_examples=[{"query": "similar question", "reply": "example reply", "score": 0.9}],
        style_card="- usually lowercase\n- concise",
    )

    assert gold == "gold reply"
    assert "Personal texting style guide:\n- usually lowercase\n- concise" in prompt
    assert prompt.index(DEMO_HEADER) < prompt.index("similar question")
    assert prompt.index("similar question") < prompt.index("example reply")
    assert prompt.index("example reply") < prompt.index(DEMO_BOUNDARY)
    assert prompt.index(DEMO_BOUNDARY) < prompt.index("current question")
    assert prompt.index("example reply") < prompt.index("current question")
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "gold reply" not in prompt


def test_personalized_prompt_renders_retrieved_multi_turn_context() -> None:
    messages = [
        {"role": "user", "content": "current question"},
        {"role": "assistant", "content": "gold reply"},
    ]
    retrieved = [
        {
            "reply": "like my retainer got loose",
            "score": 0.8,
            "context_messages": [
                {"role": "assistant", "content": "notion is good for medical stuff"},
                {"role": "user", "content": "wait wdym"},
            ],
        }
    ]

    prompt, _ = build_personalized_prompt(messages, 1, retrieved_examples=retrieved)

    # The demonstration shows the retrieved context as real turns before the reply.
    assert prompt.index(DEMO_HEADER) < prompt.index("notion is good for medical stuff")
    assert prompt.index("notion is good for medical stuff") < prompt.index("wait wdym")
    assert prompt.index("wait wdym") < prompt.index("like my retainer got loose")
    assert prompt.index("like my retainer got loose") < prompt.index(DEMO_BOUNDARY)
    assert (
        "<|im_start|>assistant\nnotion is good for medical stuff<|im_end|>\n"
        "<|im_start|>user\nwait wdym<|im_end|>\n"
        "<|im_start|>assistant\nlike my retainer got loose<|im_end|>\n"
    ) in prompt


def test_target_selection_spans_context_range_and_preserves_session_id() -> None:
    messages = []
    supervised_indexes = []
    for index in range(10):
        messages.append({"role": "user", "content": f"question {index}"})
        supervised_indexes.append(len(messages))
        messages.append({"role": "assistant", "content": f"reply {index}"})
    record = {
        "format": SFT_FORMAT,
        "example_id": "display-id",
        "session_id": "actual-session-id",
        "messages": messages,
        "supervised_indexes": supervised_indexes,
    }

    targets = _select_targets(
        [record],
        samples=8,
        seed=42,
        min_context_turns=1,
        require_preceding_user=True,
    )

    assert len(targets) == 8
    assert targets[0]["context_turns"] == 1
    assert targets[-1]["context_turns"] == 19
    assert {target["session_id"] for target in targets} == {"actual-session-id"}
