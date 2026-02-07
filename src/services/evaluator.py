"""Aggregates per-answer scores and produces final feedback report."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnswerEvaluation:
    """One Q&A evaluation."""
    question: str
    answer: str
    technical_depth: int
    clarity: int
    originality: int
    understanding: int
    feedback_snippet: str


@dataclass
class FeedbackReport:
    """Final score and feedback report."""
    overall_score: float  # 0-100
    technical_depth: float
    clarity: float
    originality: float
    understanding: float
    summary: str
    per_answer: list[AnswerEvaluation] = field(default_factory=list)


class EvaluatorService:
    """Compute overall scores and generate feedback report."""

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
                technical_depth=eval_data.get("technical_depth", 5),
                clarity=eval_data.get("clarity", 5),
                originality=eval_data.get("originality", 5),
                understanding=eval_data.get("understanding", 5),
                feedback_snippet=eval_data.get("feedback_snippet", ""),
            )
        )

    def finalize_report(self, report: FeedbackReport) -> FeedbackReport:
        """Compute overall scores (average of per-answer) and summary."""
        if not report.per_answer:
            report.overall_score = 0.0
            report.technical_depth = 0.0
            report.clarity = 0.0
            report.originality = 0.0
            report.understanding = 0.0
            report.summary = "No answers were evaluated."
            return report

        n = len(report.per_answer)
        report.technical_depth = sum(a.technical_depth for a in report.per_answer) / n
        report.clarity = sum(a.clarity for a in report.per_answer) / n
        report.originality = sum(a.originality for a in report.per_answer) / n
        report.understanding = sum(a.understanding for a in report.per_answer) / n
        report.overall_score = (
            report.technical_depth + report.clarity + report.originality + report.understanding
        ) / 4.0 * 10.0  # scale to 0-100
        report.summary = self._generate_summary(report)
        return report

    def _generate_summary(self, report: FeedbackReport) -> str:
        """Short overall summary from criteria."""
        parts = []
        if report.technical_depth >= 7:
            parts.append("Strong technical depth.")
        elif report.technical_depth >= 5:
            parts.append("Adequate technical depth; consider going deeper on implementation.")
        else:
            parts.append("Technical depth could be improved with more detail on design and code.")

        if report.clarity >= 7:
            parts.append("Explanations were clear.")
        elif report.clarity >= 5:
            parts.append("Clarity was acceptable; structure explanations more.")
        else:
            parts.append("Try to explain step-by-step for better clarity.")

        if report.originality >= 7:
            parts.append("Notable originality in approach or ideas.")
        elif report.originality >= 5:
            parts.append("Some original elements; highlight what makes your project unique.")
        else:
            parts.append("Consider emphasizing what is novel or creative in your work.")

        if report.understanding >= 7:
            parts.append("Good grasp of the implementation.")
        elif report.understanding >= 5:
            parts.append("Reasonable understanding; reinforce how parts connect.")
        else:
            parts.append("Review the implementation to strengthen your understanding.")

        return " ".join(parts)

    def save_report(self, report: FeedbackReport, path: Path) -> None:
        """Write report to HTML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._report_to_html(report)
        path.write_text(html, encoding="utf-8")

    def _report_to_html(self, report: FeedbackReport) -> str:
        """Render report as HTML."""
        rows = ""
        for a in report.per_answer:
            rows += f"""
            <tr>
                <td>{_esc(a.question)}</td>
                <td>{_esc(a.answer[:200] + ("..." if len(a.answer) > 200 else ""))}</td>
                <td>{a.technical_depth}</td>
                <td>{a.clarity}</td>
                <td>{a.originality}</td>
                <td>{a.understanding}</td>
                <td>{_esc(a.feedback_snippet)}</td>
            </tr>"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Interview Feedback Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ color: #1a1a2e; }}
    .scores {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
    .score-box {{ background: #eee; padding: 0.75rem 1rem; border-radius: 8px; min-width: 120px; }}
    .score-box strong {{ display: block; font-size: 1.5rem; color: #16213e; }}
    .summary {{ background: #f0f4ff; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
    th {{ background: #1a1a2e; color: #fff; }}
  </style>
</head>
<body>
  <h1>Interview Feedback Report</h1>
  <p><strong>Overall Score: {report.overall_score:.1f}/100</strong></p>
  <div class="scores">
    <div class="score-box">Technical depth <strong>{report.technical_depth:.1f}/10</strong></div>
    <div class="score-box">Clarity <strong>{report.clarity:.1f}/10</strong></div>
    <div class="score-box">Originality <strong>{report.originality:.1f}/10</strong></div>
    <div class="score-box">Understanding <strong>{report.understanding:.1f}/10</strong></div>
  </div>
  <div class="summary"><strong>Summary</strong><br>{_esc(report.summary)}</div>
  <h2>Per-question evaluation</h2>
  <table>
    <thead>
      <tr>
        <th>Question</th>
        <th>Answer (excerpt)</th>
        <th>Tech</th>
        <th>Clarity</th>
        <th>Originality</th>
        <th>Understanding</th>
        <th>Feedback</th>
      </tr>
    </thead>
    <tbody>{rows}
    </tbody>
  </table>
</body>
</html>"""


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
