"""Aggregates per-answer scores and produces the final interview feedback report."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnswerEvaluation:
    """One Q&A evaluation."""

    question: str
    answer: str
    communication: int
    technical_relevance: int
    confidence: int
    clarity: int
    overall_quality: int
    feedback_snippet: str


@dataclass
class FeedbackReport:
    """Final score and feedback report."""

    overall_score: float  # 0-100
    communication: float
    technical_relevance: float
    confidence: float
    clarity: float
    overall_quality: float
    summary: str
    per_answer: list[AnswerEvaluation] = field(default_factory=list)


class EvaluatorService:
    """Compute aggregated scores and generate the final report."""

    def add_evaluation(
        self,
        report: FeedbackReport,
        question: str,
        answer: str,
        eval_data: dict[str, Any],
    ) -> None:
        """Append one answer evaluation to the report."""
        report.per_answer.append(
            AnswerEvaluation(
                question=question,
                answer=answer,
                communication=_score(eval_data, "communication"),
                technical_relevance=_score(eval_data, "technical_relevance"),
                confidence=_score(eval_data, "confidence"),
                clarity=_score(eval_data, "clarity"),
                overall_quality=_score(eval_data, "overall_quality"),
                feedback_snippet=str(eval_data.get("feedback_snippet", "")).strip(),
            )
        )

    def finalize_report(self, report: FeedbackReport) -> FeedbackReport:
        """Compute overall scores and summary."""
        if not report.per_answer:
            report.overall_score = 0.0
            report.communication = 0.0
            report.technical_relevance = 0.0
            report.confidence = 0.0
            report.clarity = 0.0
            report.overall_quality = 0.0
            report.summary = "No answers were evaluated."
            return report

        n = len(report.per_answer)
        report.communication = sum(a.communication for a in report.per_answer) / n
        report.technical_relevance = sum(a.technical_relevance for a in report.per_answer) / n
        report.confidence = sum(a.confidence for a in report.per_answer) / n
        report.clarity = sum(a.clarity for a in report.per_answer) / n
        report.overall_quality = sum(a.overall_quality for a in report.per_answer) / n
        report.overall_score = (
            report.communication
            + report.technical_relevance
            + report.confidence
            + report.clarity
            + report.overall_quality
        ) / 5.0 * 10.0
        report.summary = self._generate_summary(report)
        return report

    def _generate_summary(self, report: FeedbackReport) -> str:
        """Generate a compact narrative summary from the configured criteria."""
        parts: list[str] = []
        parts.append(_summary_line("Communication", report.communication, "Keep answers more conversational and better structured."))
        parts.append(_summary_line("Technical relevance", report.technical_relevance, "Tie each answer more directly to the tools, decisions, and outcomes from the resume projects."))
        parts.append(_summary_line("Confidence", report.confidence, "Speak more decisively about your role, trade-offs, and impact."))
        parts.append(_summary_line("Clarity", report.clarity, "Use simpler step-by-step explanations with concrete examples."))
        parts.append(_summary_line("Overall quality", report.overall_quality, "Improve completeness by combining context, implementation detail, and outcomes in each answer."))
        return " ".join(parts)

    def save_report(self, report: FeedbackReport, path: Path) -> None:
        """Write report to HTML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._report_to_html(report), encoding="utf-8")

    def _report_to_html(self, report: FeedbackReport) -> str:
        rows = ""
        for answer in report.per_answer:
            rows += f"""
            <tr>
                <td>{_esc(answer.question)}</td>
                <td>{_esc(answer.answer[:220] + ("..." if len(answer.answer) > 220 else ""))}</td>
                <td>{answer.communication}</td>
                <td>{answer.technical_relevance}</td>
                <td>{answer.confidence}</td>
                <td>{answer.clarity}</td>
                <td>{answer.overall_quality}</td>
                <td>{_esc(answer.feedback_snippet)}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Interview Feedback Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #102030; }}
    h1 {{ margin-bottom: 0.5rem; }}
    .scores {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.25rem 0; }}
    .score-box {{ background: #f3f6fb; border: 1px solid #d8e0ee; padding: 0.85rem 1rem; border-radius: 12px; min-width: 150px; }}
    .score-box strong {{ display: block; font-size: 1.6rem; color: #164e63; }}
    .summary {{ background: #eef6ff; border: 1px solid #c9def3; padding: 1rem; border-radius: 12px; margin: 1rem 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #d6dce7; padding: 0.6rem; text-align: left; vertical-align: top; }}
    th {{ background: #16324f; color: #fff; }}
  </style>
</head>
<body>
  <h1>Interview Feedback Report</h1>
  <p><strong>Overall Score: {report.overall_score:.1f}/100</strong></p>
  <div class="scores">
    <div class="score-box">Communication <strong>{report.communication:.1f}/10</strong></div>
    <div class="score-box">Technical relevance <strong>{report.technical_relevance:.1f}/10</strong></div>
    <div class="score-box">Confidence <strong>{report.confidence:.1f}/10</strong></div>
    <div class="score-box">Clarity <strong>{report.clarity:.1f}/10</strong></div>
    <div class="score-box">Overall quality <strong>{report.overall_quality:.1f}/10</strong></div>
  </div>
  <div class="summary"><strong>Summary</strong><br>{_esc(report.summary)}</div>
  <h2>Per-question evaluation</h2>
  <table>
    <thead>
      <tr>
        <th>Question</th>
        <th>Answer (excerpt)</th>
        <th>Communication</th>
        <th>Technical relevance</th>
        <th>Confidence</th>
        <th>Clarity</th>
        <th>Overall quality</th>
        <th>Feedback</th>
      </tr>
    </thead>
    <tbody>{rows}
    </tbody>
  </table>
</body>
</html>"""


def _score(eval_data: dict[str, Any], key: str) -> int:
    try:
        return max(0, min(10, int(eval_data.get(key, 5))))
    except Exception:
        return 5


def _summary_line(label: str, value: float, improvement_line: str) -> str:
    if value >= 8:
        return f"{label} is a clear strength."
    if value >= 6:
        return f"{label} is solid overall."
    return improvement_line


def _esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
