"""LLM service using Ollama for resume extraction, question generation, and answer evaluation."""
import json
from typing import Any

import httpx

try:
    import ollama
except ImportError:
    ollama = None


class LLMService:
    """Generate structured interview content via Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _ensure_client(self):
        if ollama is None:
            raise RuntimeError("ollama not installed. pip install ollama")
        return ollama

    def is_available(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def extract_resume_profile(self, resume_text: str) -> dict[str, Any]:
        """Extract structured candidate information from resume text."""
        sys_prompt = """You extract resume data into structured JSON.
Return JSON only with this exact structure:
{
  "candidate_name": "string",
  "headline": "string",
  "professional_summary": "string",
  "skills": ["string"],
  "projects": [
    {
      "name": "string",
      "role": "string",
      "summary": "string",
      "skills_used": ["string"]
    }
  ],
  "work_experience": [
    {
      "company": "string",
      "role": "string",
      "duration": "string",
      "highlights": ["string"]
    }
  ]
}
Rules:
- Use empty strings or empty arrays when information is missing.
- Include every meaningful project mentioned in the resume.
- Normalize skills into concise technology names.
- Do not add markdown or commentary."""

        content = f"Resume text:\n{resume_text[:18000]}\n\nReturn JSON only."
        try:
            client = self._ensure_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": content},
                ],
            )
            text = _message_text(response)
            return _normalize_resume_profile(_parse_json(text))
        except Exception:
            return _fallback_resume_profile(resume_text)

    def generate_resume_question(
        self,
        resume_profile: dict[str, Any],
        previous_questions: list[str],
        answers: list[str],
        question_number: int,
    ) -> str:
        """Generate the next interview question from resume skills and projects."""
        sys_prompt = """You are a technical interviewer.
Ask exactly one concise interview question at a time.
Base questions on the candidate's resume, especially:
- skills they have worked on
- projects across their work experience
- implementation decisions, challenges, trade-offs, ownership, and outcomes
Avoid repeating prior questions.
Keep the question answerable in spoken or typed interview style.
Output only the question."""

        payload = {
            "resume_profile": resume_profile,
            "previous_questions": previous_questions,
            "answers_so_far": answers,
            "question_number": question_number,
        }
        try:
            client = self._ensure_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                ],
            )
            text = _message_text(response)
            return _clean_question(text) or self._fallback_question(
                resume_profile,
                previous_questions,
                question_number,
            )
        except Exception:
            return self._fallback_question(resume_profile, previous_questions, question_number)

    def evaluate_interview_answer(
        self,
        question: str,
        answer: str,
        resume_profile: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate an answer on the required report dimensions."""
        sys_prompt = """You evaluate interview answers.
Return JSON only with this exact structure:
{
  "communication": <0-10 integer>,
  "technical_relevance": <0-10 integer>,
  "confidence": <0-10 integer>,
  "clarity": <0-10 integer>,
  "overall_quality": <0-10 integer>,
  "feedback_snippet": "one short sentence"
}
Scoring guidance:
- communication: conversational effectiveness and completeness
- technical_relevance: alignment to the asked question and resume/project context
- confidence: decisiveness and ownership shown in the answer
- clarity: structure and understandability
- overall_quality: overall usefulness of the answer for interview assessment"""
        payload = {
            "question": question,
            "answer": answer,
            "resume_profile": resume_profile,
        }
        try:
            client = self._ensure_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                ],
            )
            data = _parse_json(_message_text(response))
            return {
                "communication": _bounded_int(data.get("communication")),
                "technical_relevance": _bounded_int(data.get("technical_relevance")),
                "confidence": _bounded_int(data.get("confidence")),
                "clarity": _bounded_int(data.get("clarity")),
                "overall_quality": _bounded_int(data.get("overall_quality")),
                "feedback_snippet": str(data.get("feedback_snippet", "Answer evaluated successfully.")).strip(),
            }
        except Exception:
            return {
                "communication": 5,
                "technical_relevance": 5,
                "confidence": 5,
                "clarity": 5,
                "overall_quality": 5,
                "feedback_snippet": "Evaluation fallback was used.",
            }

    def _fallback_question(
        self,
        resume_profile: dict[str, Any],
        previous_questions: list[str],
        question_number: int,
    ) -> str:
        projects = resume_profile.get("projects") or []
        skills = resume_profile.get("skills") or []
        project_name = ""
        if projects:
            project_name = str(projects[min(question_number - 1, len(projects) - 1)].get("name", "")).strip()
        if project_name:
            candidate = f"Can you walk me through your work on {project_name} and the key technical decisions you made?"
        elif skills:
            skill = str(skills[min(question_number - 1, len(skills) - 1)]).strip()
            candidate = f"How have you used {skill} in your past projects, and what challenges did you solve with it?"
        else:
            candidate = "Can you describe one important project from your experience and explain your technical contribution?"
        if candidate in previous_questions:
            return "Which project best demonstrates your strongest technical skills, and what was your specific contribution?"
        return candidate


def _message_text(response: dict[str, Any]) -> str:
    msg = response.get("message", {}) or {}
    return (msg.get("content") or "").strip()


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        for part in cleaned.split("```"):
            part = part.strip()
            if part.startswith("json"):
                cleaned = part[4:].strip()
                break
            if part.startswith("{"):
                cleaned = part
                break
    return json.loads(cleaned)


def _normalize_resume_profile(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_name": str(data.get("candidate_name", "")).strip(),
        "headline": str(data.get("headline", "")).strip(),
        "professional_summary": str(data.get("professional_summary", "")).strip(),
        "skills": _string_list(data.get("skills")),
        "projects": [
            {
                "name": str(project.get("name", "")).strip(),
                "role": str(project.get("role", "")).strip(),
                "summary": str(project.get("summary", "")).strip(),
                "skills_used": _string_list(project.get("skills_used")),
            }
            for project in (data.get("projects") or [])
            if isinstance(project, dict)
        ],
        "work_experience": [
            {
                "company": str(item.get("company", "")).strip(),
                "role": str(item.get("role", "")).strip(),
                "duration": str(item.get("duration", "")).strip(),
                "highlights": _string_list(item.get("highlights")),
            }
            for item in (data.get("work_experience") or [])
            if isinstance(item, dict)
        ],
    }


def _fallback_resume_profile(resume_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    skills = []
    for line in lines:
        lower = line.lower()
        if "skill" in lower or "technology" in lower or "tools" in lower:
            _, _, tail = line.partition(":")
            skills.extend(part.strip(" -") for part in tail.split(",") if part.strip())
    return {
        "candidate_name": lines[0] if lines else "",
        "headline": lines[1] if len(lines) > 1 else "",
        "professional_summary": " ".join(lines[2:6])[:500],
        "skills": skills[:12],
        "projects": [],
        "work_experience": [],
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _clean_question(text: str) -> str:
    question = text.strip()
    for prefix in ("Question:", "Q:", "1.", "1)", "-", "*"):
        if question.lower().startswith(prefix.lower()):
            question = question[len(prefix):].strip()
    return question.replace("\n", " ").strip()


def _bounded_int(value: Any) -> int:
    try:
        return max(0, min(10, int(value)))
    except Exception:
        return 5
