from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def _esc(escape_text: Optional[Callable[[str], str]], value: Any) -> str:
    text = "" if value is None else str(value)
    return escape_text(text) if escape_text else text


def build_deliberation_summary_html(
    *,
    question: str,
    answers: Dict[str, str],
    winner: str,
    details: Dict[str, Any],
    tally: Dict[str, Any],
    colors: Dict[str, str],
    short_id: Callable[[str], str],
    safe_markdown_html: Callable[[str], str],
    escape_text: Optional[Callable[[str], str]],
    normalize_rubric_weights: Callable[[Optional[Dict[str, Any]]], Dict[str, int]],
) -> str:
    valid_votes = details.get("valid_votes", {}) or {}
    invalid_votes = details.get("invalid_votes", {}) or {}
    vote_messages = details.get("vote_messages", {}) or {}
    voters_used = details.get("voters_used", []) or []
    timings_ms = details.get("timings_ms", {}) or {}
    rubric_weights = normalize_rubric_weights(details.get("rubric_weights"))

    ranking_rows = "".join(
        f"<li><strong>{_esc(escape_text, short_id(model_id))}</strong> · score {tally[model_id]}</li>"
        for model_id in sorted(tally.keys(), key=lambda key: (-tally[key], key))
    ) or "<li>No scores available.</li>"

    latency_rows = "".join(
        f"<li><strong>{_esc(escape_text, short_id(model_id))}</strong> · {int(timings_ms[model_id])} ms</li>"
        for model_id in sorted(timings_ms.keys(), key=lambda mid: (timings_ms[mid], mid))
    ) if timings_ms else ""

    note_rows = "".join(
        f"<li><strong>{_esc(escape_text, short_id(voter))}</strong> · {_esc(escape_text, msg)}</li>"
        for voter, msg in vote_messages.items()
    ) if vote_messages else ""

    winner_answer_html = safe_markdown_html(answers.get(winner, ""))
    mid_color = colors["text_secondary"]
    border_color = colors["border"]
    alt_color = colors["panel_subtle"]

    html = [
        "<div style='font-size:20px; font-weight:700; margin-bottom:6px;'>Council Summary</div>",
        f"<div style='color:{mid_color}; margin-bottom:16px;'>{_esc(escape_text, question)}</div>",
        "<table style='width:100%; border-collapse:separate; border-spacing:0 8px; margin-bottom:18px;'>",
        "<tr>",
        f"<td style='padding:12px; border:1px solid {border_color}; border-radius:12px; background:{alt_color};'><strong>Winner</strong><br>{_esc(escape_text, short_id(winner))}</td>",
        f"<td style='padding:12px; border:1px solid {border_color}; border-radius:12px; background:{alt_color};'><strong>Valid ballots</strong><br>{len(valid_votes)}</td>",
        f"<td style='padding:12px; border:1px solid {border_color}; border-radius:12px; background:{alt_color};'><strong>Invalid ballots</strong><br>{len(invalid_votes)}</td>",
        f"<td style='padding:12px; border:1px solid {border_color}; border-radius:12px; background:{alt_color};'><strong>Voters used</strong><br>{_esc(escape_text, ', '.join(short_id(v) for v in voters_used)) if voters_used else 'All selected models'}</td>",
        "</tr>",
        "</table>",
        f"<div style='font-weight:700; margin-bottom:6px;'>Rubric weights</div><div style='margin-bottom:16px; color:{mid_color};'>{_esc(escape_text, rubric_weights)}</div>",
        "<div style='font-weight:700; margin-bottom:6px;'>Scoreboard</div>",
        f"<ul style='margin-top:0; margin-bottom:16px;'>{ranking_rows}</ul>",
    ]
    if latency_rows:
        html.extend([
            "<div style='font-weight:700; margin-bottom:6px;'>Answer latencies</div>",
            f"<ul style='margin-top:0; margin-bottom:16px;'>{latency_rows}</ul>",
        ])
    if note_rows:
        html.extend([
            "<div style='font-weight:700; margin-bottom:6px;'>Ballot notes</div>",
            f"<ul style='margin-top:0; margin-bottom:16px;'>{note_rows}</ul>",
        ])
    html.extend([
        "<div style='font-weight:700; margin-bottom:6px;'>Winning answer</div>",
        f"<div style='padding:14px; border:1px solid {border_color}; border-radius:14px;'>{winner_answer_html}</div>",
    ])
    return "".join(html)


def build_winner_html(
    *,
    winner: str,
    answer: str,
    tally: Dict[str, Any],
    details: Dict[str, Any],
    colors: Dict[str, str],
    short_id: Callable[[str], str],
    safe_markdown_html: Callable[[str], str],
    escape_text: Optional[Callable[[str], str]],
) -> str:
    timings_ms = details.get("timings_ms", {}) or {}
    timing = timings_ms.get(winner)
    meta = [f"Score {tally.get(winner, 0)}"]
    if isinstance(timing, (int, float)):
        meta.append(f"{int(timing)} ms")
    return f"""
    <div style="font-size:22px; font-weight:800; margin-bottom:6px;">{_esc(escape_text, short_id(winner))}</div>
    <div style="margin-bottom:14px; color:{colors['text_secondary']};">{' · '.join(meta)}</div>
    {safe_markdown_html(answer or "_No answer available._")}
    """


def build_ballots_html(
    *,
    answers: Dict[str, str],
    details: Dict[str, Any],
    tally: Dict[str, Any],
    colors: Dict[str, str],
    short_id: Callable[[str], str],
    escape_text: Optional[Callable[[str], str]],
    normalize_rubric_weights: Callable[[Optional[Dict[str, Any]]], Dict[str, int]],
) -> str:
    valid_votes = details.get("valid_votes", {}) or {}
    invalid_votes = details.get("invalid_votes", {}) or {}
    vote_messages = details.get("vote_messages", {}) or {}
    rubric_weights = normalize_rubric_weights(details.get("rubric_weights"))
    sections = [
        "<div style='font-size:20px; font-weight:700; margin-bottom:8px;'>Ballots & Voting</div>",
        f"<div style='margin-bottom:12px; color:{colors['text_secondary']};'>Rubric: {_esc(escape_text, rubric_weights)}</div>",
    ]
    if tally:
        sections.append("<div style='font-weight:700; margin-bottom:6px;'>Totals</div>")
        sections.append("<ul style='margin-top:0; margin-bottom:16px;'>" + "".join(
            f"<li><strong>{_esc(escape_text, short_id(model_id))}</strong> · {_esc(escape_text, tally[model_id])}</li>"
            for model_id in sorted(tally.keys(), key=lambda key: (-tally[key], key))
        ) + "</ul>")
    if valid_votes:
        sections.append("<div style='font-weight:700; margin-bottom:6px;'>Valid ballots</div><ul style='margin-top:0; margin-bottom:16px;'>")
        for voter, ballot in valid_votes.items():
            ranked = []
            for candidate in answers.keys():
                candidate_scores = ballot.get("scores", {}).get(candidate)
                if candidate_scores:
                    weighted = sum(candidate_scores[k] * rubric_weights[k] for k in rubric_weights.keys())
                    ranked.append(f"{_esc(escape_text, short_id(candidate))}: {weighted}")
            sections.append(f"<li><strong>{_esc(escape_text, short_id(voter))}</strong> · {'; '.join(ranked)}</li>")
        sections.append("</ul>")
    if invalid_votes:
        sections.append("<div style='font-weight:700; margin-bottom:6px;'>Invalid ballots</div><ul style='margin-top:0; margin-bottom:16px;'>")
        for voter, message in invalid_votes.items():
            sections.append(f"<li><strong>{_esc(escape_text, short_id(voter))}</strong> · {_esc(escape_text, message)}</li>")
        sections.append("</ul>")
    if vote_messages:
        sections.append("<div style='font-weight:700; margin-bottom:6px;'>Notes</div><ul style='margin-top:0;'>")
        for voter, message in vote_messages.items():
            sections.append(f"<li><strong>{_esc(escape_text, short_id(voter))}</strong> · {_esc(escape_text, message)}</li>")
        sections.append("</ul>")
    return "".join(sections)


def build_discussion_transcript_html(
    *,
    transcript: List[Dict[str, Any]],
    colors: Dict[str, str],
    safe_markdown_html: Callable[[str], str],
    placeholder_html: Callable[[str, str], str],
    escape_text: Optional[Callable[[str], str]],
) -> str:
    if not transcript:
        return placeholder_html("No Transcript Yet", "Discussion turns will stream here as each agent responds.")
    border_color = colors["border"]
    alt_color = colors["panel_subtle"]
    parts = []
    for entry in transcript:
        agent = _esc(escape_text, entry.get("agent", "Unknown"))
        persona = entry.get("persona")
        meta = f"{agent} [{_esc(escape_text, persona)}]" if persona else agent
        parts.append(
            f"""
            <div style="margin-bottom:10px; padding:12px; border:1px solid {border_color}; border-radius:12px; background:{alt_color};">
                <div style="font-weight:700; margin-bottom:6px;">Turn {entry.get('turn', 0)} · {meta}</div>
                <div>{safe_markdown_html(entry.get('message', ''))}</div>
            </div>
            """
        )
    return "".join(parts)


def build_discussion_report_html(
    *,
    question: str,
    transcript: List[Dict[str, Any]],
    synthesis: Optional[str],
    colors: Dict[str, str],
    safe_markdown_html: Callable[[str], str],
    placeholder_html: Callable[[str, str], str],
    escape_text: Optional[Callable[[str], str]],
) -> str:
    border_color = colors["border"]
    alt_color = colors["panel_subtle"]
    transcript_html = build_discussion_transcript_html(
        transcript=transcript,
        colors=colors,
        safe_markdown_html=safe_markdown_html,
        placeholder_html=placeholder_html,
        escape_text=escape_text,
    )
    synthesis_html = safe_markdown_html(synthesis if synthesis and synthesis.strip() else "Synthesis not available.")
    return f"""
    <div style="font-size:20px; font-weight:700; margin-bottom:6px;">Discussion Report</div>
    <div style="margin-bottom:14px; color:{colors['text_secondary']};">{_esc(escape_text, question)}</div>
    <div style="padding:12px; border:1px solid {border_color}; border-radius:12px; background:{alt_color}; margin-bottom:16px;">
        <strong>Turns recorded</strong><br>{len(transcript)}
    </div>
    <div style="font-weight:700; margin-bottom:6px;">Final synthesis</div>
    <div style="padding:14px; border:1px solid {border_color}; border-radius:14px; margin-bottom:16px;">{synthesis_html}</div>
    <div style="font-weight:700; margin-bottom:6px;">Transcript</div>
    {transcript_html}
    """
