# Embedded HTML and CSS are kept readable instead of wrapped as Python source.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from imessage_mlx.utils import atomic_write_text

METHODS = (
    ("base", "Base model", "Qwen with only the conversation"),
    ("retrieval", "Retrieval", "Qwen plus four similar past exchanges"),
    (
        "retrieval_style",
        "Retrieval + style",
        "Retrieval plus the generated texting-style guide",
    ),
)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _score(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, int | float) else "—"


def _summary(examples: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, int]]:
    averages: dict[str, float] = {}
    wins = {key: 0 for key, _, _ in METHODS}
    for key, _, _ in METHODS:
        values = [
            float(example.get("content_cosine", {}).get(key))
            for example in examples
            if isinstance(example.get("content_cosine", {}).get(key), int | float)
        ]
        averages[key] = sum(values) / len(values) if values else 0.0

    for example in examples:
        scores = example.get("content_cosine", {})
        available = {
            key: float(scores[key])
            for key, _, _ in METHODS
            if isinstance(scores.get(key), int | float)
        }
        if available:
            wins[max(available, key=available.get)] += 1
    return averages, wins


def _context_html(context: list[dict[str, Any]]) -> str:
    bubbles = []
    for turn in context:
        mine = turn.get("role") == "you"
        role = "You" if mine else "Them"
        css_class = "mine" if mine else "theirs"
        bubbles.append(
            f'<div class="bubble-row {css_class}">'
            f'<div class="bubble"><span class="speaker">{role}</span>'
            f'<div class="message">{_escape(turn.get("content", ""))}</div></div></div>'
        )
    return "".join(bubbles)


def _retrieved_context_html(item: dict[str, Any]) -> str:
    turns = item.get("context_messages") or []
    if not turns:
        return _escape(item.get("context_query") or item.get("query", ""))
    bubbles = []
    for turn in turns:
        mine = turn.get("role") == "assistant"
        role = "You" if mine else "Them"
        css_class = "mine" if mine else "theirs"
        bubbles.append(
            f'<div class="bubble-row {css_class}">'
            f'<div class="bubble"><span class="speaker">{role}</span>'
            f'<div class="message">{_escape(turn.get("content", ""))}</div></div></div>'
        )
    return "".join(bubbles)


def _retrieval_html(retrieved: list[dict[str, Any]]) -> str:
    rows = []
    for item in retrieved:
        rows.append(
            '<div class="retrieved-pair">'
            f'<div><span class="small-label">Conversation</span>'
            f'<div class="conversation">{_retrieved_context_html(item)}</div></div>'
            f'<div><span class="small-label">Your reply</span>{_escape(item.get("reply", ""))}</div>'
            f'<div class="similarity">Similarity {_score(item.get("score"))}</div>'
            "</div>"
        )
    return "".join(rows)


def _example_html(example: dict[str, Any], number: int) -> str:
    scores = example.get("content_cosine", {})
    available = {
        key: float(scores[key]) for key, _, _ in METHODS if isinstance(scores.get(key), int | float)
    }
    winner = max(available, key=available.get) if available else None
    candidates = []
    for key, label, description in METHODS:
        badge = '<span class="badge">Highest content score</span>' if key == winner else ""
        candidates.append(
            '<section class="candidate">'
            f'<div class="candidate-heading"><div><h3>{label}</h3>'
            f"<p>{description}</p></div>{badge}</div>"
            f'<div class="candidate-reply">{_escape(example.get(key, ""))}</div>'
            f'<div class="candidate-score">Content similarity: {_score(scores.get(key))}</div>'
            "</section>"
        )

    return (
        '<article class="example">'
        f'<div class="example-number">Example {number}</div>'
        '<div class="section-label">Conversation before your reply</div>'
        f'<div class="conversation">{_context_html(example.get("context", []))}</div>'
        '<div class="actual">'
        '<div><span class="section-label">What you actually sent</span>'
        f'<div class="actual-reply">{_escape(example.get("gold", ""))}</div></div>'
        f'<div class="query"><span class="section-label">Context used for search</span>'
        f'<div class="message">{_escape(example.get("retrieval_query", example.get("query", "")))}</div></div></div>'
        f'<div class="candidates">{"".join(candidates)}</div>'
        "<details><summary>See the four past exchanges retrieval used</summary>"
        f'<div class="retrieved-list">{_retrieval_html(example.get("retrieved", []))}</div>'
        "</details>"
        "</article>"
    )


def render_personalization_report(data: dict[str, Any]) -> str:
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Personalization results must contain at least one example")
    averages, wins = _summary(examples)
    best_average = max(averages, key=averages.get)
    method_names = {key: label for key, label, _ in METHODS}
    score_cards = "".join(
        '<div class="stat">'
        f"<span>{label}</span><strong>{averages[key]:.3f}</strong>"
        f"<small>highest in {wins[key]} of {len(examples)}</small></div>"
        for key, label, _ in METHODS
    )
    example_sections = "".join(
        _example_html(example, index) for index, example in enumerate(examples, start=1)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Personal texting experiment</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f6f7f9; --surface: #ffffff; --surface-2: #f0f2f5;
      --text: #17191c; --muted: #656b75; --line: #dfe3e8;
      --accent: #2563eb; --accent-soft: #e8f0ff; --mine: #dceafe;
      --theirs: #eceef1; --success: #16794b;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111315; --surface: #191c1f; --surface-2: #22262a;
        --text: #f1f3f5; --muted: #a2a9b3; --line: #32373d;
        --accent: #7aa7ff; --accent-soft: #1d3154; --mine: #1f3b61;
        --theirs: #292d32; --success: #62c993;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: var(--bg); color: var(--text);
      font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ width: min(1160px, calc(100% - 32px)); margin: 0 auto; padding: 48px 0 80px; }}
    h1 {{ margin: 0; font-size: clamp(28px, 4vw, 44px); line-height: 1.1; letter-spacing: -0.03em; }}
    h2, h3, p {{ margin-top: 0; }}
    .intro {{ max-width: 760px; margin: 14px 0 28px; color: var(--muted); font-size: 17px; }}
    .result {{
      padding: 16px 18px; background: var(--accent-soft); border-radius: 12px;
      margin-bottom: 18px; color: var(--text);
    }}
    .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }}
    .stat {{ padding: 18px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }}
    .stat span, .stat small {{ display: block; color: var(--muted); }}
    .stat strong {{ display: block; font-size: 30px; margin: 3px 0; }}
    .explanation {{
      padding: 18px; border-left: 3px solid var(--accent); background: var(--surface);
      margin-bottom: 42px;
    }}
    .examples-title {{ margin: 0 0 16px; font-size: 24px; }}
    .example {{
      background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
      padding: clamp(18px, 3vw, 30px); margin-bottom: 24px;
    }}
    .example-number {{ color: var(--accent); font-weight: 700; margin-bottom: 18px; }}
    .section-label {{
      display: block; color: var(--muted); font-size: 12px; font-weight: 700;
      letter-spacing: .06em; text-transform: uppercase; margin-bottom: 7px;
    }}
    .conversation {{ max-width: 720px; padding: 18px; background: var(--surface-2); border-radius: 12px; }}
    .bubble-row {{ display: flex; margin: 7px 0; }}
    .bubble-row.mine {{ justify-content: flex-end; }}
    .bubble {{ max-width: 78%; padding: 9px 12px; background: var(--theirs); border-radius: 15px; }}
    .mine .bubble {{ background: var(--mine); }}
    .speaker {{ display: block; color: var(--muted); font-size: 11px; font-weight: 700; }}
    .message, .candidate-reply, .actual-reply {{ white-space: pre-wrap; }}
    .actual {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 22px 0;
      padding: 16px 18px; border: 2px solid var(--accent); border-radius: 12px;
    }}
    .actual-reply {{ font-size: 19px; font-weight: 650; }}
    .query {{ color: var(--muted); }}
    .candidates {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .candidate {{ padding: 16px; border: 1px solid var(--line); border-radius: 12px; min-width: 0; }}
    .candidate-heading {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; }}
    .candidate h3 {{ margin-bottom: 3px; font-size: 16px; }}
    .candidate p {{ color: var(--muted); font-size: 12px; }}
    .candidate-reply {{ min-height: 70px; margin: 18px 0; font-size: 17px; }}
    .candidate-score {{ color: var(--muted); font-size: 12px; }}
    .badge {{
      flex: none; color: var(--success); font-size: 10px; font-weight: 700;
      text-transform: uppercase; max-width: 90px; text-align: right;
    }}
    details {{ margin-top: 18px; border-top: 1px solid var(--line); padding-top: 14px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 650; }}
    .retrieved-list {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 14px; }}
    .retrieved-pair {{ position: relative; padding: 13px; background: var(--surface-2); border-radius: 10px; }}
    .retrieved-pair > div {{ margin-bottom: 8px; }}
    .small-label {{ display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; }}
    .similarity {{ color: var(--muted); font-size: 11px; }}
    footer {{ color: var(--muted); padding-top: 20px; }}
    @media (max-width: 800px) {{
      .stats, .candidates, .actual, .retrieved-list {{ grid-template-columns: 1fr; }}
      .candidate-reply {{ min-height: auto; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Personal texting experiment</h1>
    <p class="intro">Eight real conversations were hidden from the model. This report compares
    its plain reply, a reply informed by similar past texts, and one that also uses your style guide.</p>
  </header>
  <div class="result"><strong>Result:</strong> {_escape(method_names[best_average])} had the
  highest average content similarity in this small run. Read the conversations below—human judgment
  matters more than this automatic score.</div>
  <div class="stats">{score_cards}</div>
  <div class="explanation"><strong>How to read the score:</strong> higher means the generated reply
  is closer in meaning to what you actually sent. It does not measure whether the wording sounds
  like you, and it is not a confidence percentage.</div>
  <h2 class="examples-title">Conversation-by-conversation comparison</h2>
  {example_sections}
  <footer>Private local report · Model: {_escape(data.get("model_name", "unknown"))} ·
  Retrieval: {_escape(data.get("embedding_model", "unknown"))}</footer>
</main>
</body>
</html>
"""


def render_personalization_report_file(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source = Path(input_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Personalization results must be a JSON object")
    report = render_personalization_report(data)
    destination = atomic_write_text(output_path, report)
    return {
        "input_path": str(source),
        "output_path": str(destination),
        "examples": len(data.get("examples", [])),
    }


def _stamp_method_names(data: dict[str, Any]) -> list[str]:
    configured = data.get("methods")
    if isinstance(configured, list):
        return [str(value) for value in configured]
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        return [str(value) for value in metrics]
    examples = data.get("examples")
    if isinstance(examples, list) and examples:
        outputs = examples[0].get("outputs")
        if isinstance(outputs, dict):
            return [str(value) for value in outputs]
    return []


def render_stamp_report(data: dict[str, Any]) -> str:
    """Render a private, self-contained style-transfer comparison report."""
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("STAMP results must contain at least one example")
    methods = _stamp_method_names(data)
    if not methods:
        raise ValueError("STAMP results must name at least one evaluated method")

    aggregate = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    metric_names = ("style_probability", "semantic_similarity", "fluency", "reward")
    summary_rows = []
    for method in methods:
        values = aggregate.get(method, {}) if isinstance(aggregate, dict) else {}
        cells = "".join(
            f"<td>{_score(values.get(metric))}</td>" if isinstance(values, dict) else "<td>—</td>"
            for metric in metric_names
        )
        summary_rows.append(f"<tr><th>{_escape(method)}</th>{cells}</tr>")

    rendered_examples = []
    for index, example in enumerate(examples, start=1):
        outputs = example.get("outputs") if isinstance(example.get("outputs"), dict) else {}
        per_method = (
            example.get("metrics") if isinstance(example.get("metrics"), dict) else {}
        )
        candidates = []
        for method in methods:
            scores = per_method.get(method, {}) if isinstance(per_method, dict) else {}
            score_line = " · ".join(
                f"{metric.replace('_', ' ')} {_score(scores.get(metric))}"
                for metric in metric_names
                if isinstance(scores, dict) and metric in scores
            )
            candidates.append(
                '<section class="stamp-candidate">'
                f"<h3>{_escape(method)}</h3>"
                f'<div class="stamp-message">{_escape(outputs.get(method, ""))}</div>'
                f'<div class="stamp-scores">{_escape(score_line)}</div>'
                "</section>"
            )
        rendered_examples.append(
            '<article class="stamp-example">'
            f'<div class="example-number">Example {index}</div>'
            '<div class="stamp-pair">'
            '<section><span class="section-label">Neutral source</span>'
            f'<div class="stamp-message">{_escape(example.get("neutral", ""))}</div></section>'
            '<section><span class="section-label">Original iMessage style</span>'
            f'<div class="stamp-message gold">{_escape(example.get("target", ""))}</div></section>'
            "</div>"
            f'<div class="stamp-candidates">{"".join(candidates)}</div>'
            "</article>"
        )

    summary = data.get("classifier_metrics")
    classifier_note = ""
    if isinstance(summary, dict):
        classifier_note = (
            "<p>Style-classifier holdout: "
            + ", ".join(f"{_escape(key)} {_score(value)}" for key, value in summary.items())
            + ".</p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>iMessage STAMP experiment</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg:#f6f7f9; --surface:#fff; --surface2:#eef1f5; --text:#17191c;
      --muted:#656b75; --line:#dfe3e8; --accent:#2563eb; --gold:#e8f0ff;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg:#111315; --surface:#191c1f; --surface2:#22262a; --text:#f1f3f5;
        --muted:#a2a9b3; --line:#32373d; --accent:#7aa7ff; --gold:#1f3b61;
      }}
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text);
      font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:auto; padding:48px 0 80px; }}
    h1 {{ margin:0; font-size:clamp(28px,4vw,44px); letter-spacing:-.03em; }}
    .intro,.stamp-scores {{ color:var(--muted); }}
    .intro {{ max-width:760px; font-size:17px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--surface); margin:28px 0; }}
    th,td {{ padding:12px; border:1px solid var(--line); text-align:left; }}
    .stamp-example {{ background:var(--surface); border:1px solid var(--line);
      border-radius:16px; padding:24px; margin:24px 0; }}
    .example-number {{ color:var(--accent); font-weight:700; margin-bottom:14px; }}
    .section-label {{ display:block; color:var(--muted); font-size:11px;
      font-weight:700; letter-spacing:.06em; text-transform:uppercase; margin-bottom:7px; }}
    .stamp-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .stamp-pair section,.stamp-candidate {{ background:var(--surface2); padding:16px;
      border-radius:12px; min-width:0; }}
    .stamp-message {{ white-space:pre-wrap; font-size:17px; }}
    .stamp-message.gold {{ background:var(--gold); padding:12px; border-radius:10px; }}
    .stamp-candidates {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
      gap:12px; margin-top:12px; }}
    .stamp-candidate h3 {{ margin:0 0 12px; font-size:15px; }}
    .stamp-scores {{ margin-top:16px; font-size:11px; }}
    @media (max-width:760px) {{ .stamp-pair {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <h1>iMessage STAMP experiment</h1>
  <p class="intro">Neutral drafts are rewritten in the phone owner’s texting style.
  Automatic style scores are in-domain proxies, not proof of authorship.</p>
  {classifier_note}
  <table><thead><tr><th>Method</th><th>Style</th><th>Meaning</th>
  <th>Fluency</th><th>Composite</th></tr></thead>
  <tbody>{"".join(summary_rows)}</tbody></table>
  {"".join(rendered_examples)}
  <footer>Private local report · Run {_escape(data.get("run_name", "unknown"))}</footer>
</main></body></html>
"""


def render_stamp_report_file(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source = Path(input_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("STAMP results must be a JSON object")
    destination = atomic_write_text(output_path, render_stamp_report(data))
    return {
        "input_path": str(source),
        "output_path": str(destination),
        "examples": len(data.get("examples", [])),
    }
