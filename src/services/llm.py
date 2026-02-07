"""LLM service using Ollama (local, free) for questions and evaluation."""
import json
from typing import Any

import httpx

try:
    import ollama
except ImportError:
    ollama = None


class LLMService:
    """Generate interview questions and evaluate answers via Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = None

    def _ensure_client(self):
        if ollama is None:
            raise RuntimeError("ollama not installed. pip install ollama")
        # ollama client uses env OLLAMA_HOST or default localhost:11434
        return ollama

    def is_available(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def generate_question(
        self,
        screen_content: str,
        transcript_so_far: str,
        previous_questions: list[str],
        question_type: str = "context",
    ) -> str:
        """
        Generate one interview question from screen content and transcript.
        question_type: 'context' (first) or 'followup'
        """
        client = self._ensure_client()
        sys_prompt = """You are an expert technical interviewer. You are watching a student present their project (screen share + speech).
Your job is to ask ONE clear, concise interview question. Output ONLY the question, no preamble or numbering.
Focus on: architecture, design choices, implementation details, trade-offs, testing, or challenges.
Ask one question at a time. Be specific to what you see on screen or what they said."""

        content = f"""Screen content (OCR from their screen):\n{screen_content or '(none yet)'}\n\n"""
        content += f"Student has said so far:\n{transcript_so_far or '(nothing yet)'}\n\n"
        if previous_questions:
            content += f"Questions already asked:\n" + "\n".join(f"- {q}" for q in previous_questions[-5:]) + "\n\n"
        content += "Generate the next single interview question (one sentence, output only the question):"

        try:
            r = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": content},
                ],
            )
            msg = r.get("message", {}) or {}
            text = (msg.get("content") or "").strip()
            # trim common prefixes
            for prefix in ("Question:", "Q:", "1.", "1)", "- "):
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
            return text or "Can you walk us through the main components of your project?"
        except Exception as e:
            return f"Can you explain the main idea behind your project? (LLM error: {e})"

    def evaluate_response(
        self,
        question: str,
        answer: str,
        screen_context: str,
    ) -> dict[str, Any]:
        """
        Evaluate one Q&A pair. Returns dict with scores (0-10) and short feedback.
        Keys: technical_depth, clarity, originality, understanding, feedback_snippet
        """
        client = self._ensure_client()
        sys_prompt = """You are an expert evaluator for technical project presentations.
Score the student's answer from 0 to 10 on each criterion. Be fair and constructive.
Respond with a JSON object only, no markdown. Use this exact structure:
{"technical_depth": <0-10>, "clarity": <0-10>, "originality": <0-10>, "understanding": <0-10>, "feedback_snippet": "<one short sentence>"}"""

        user_content = f"""Question asked: {question}
Student's answer: {answer}
Relevant screen context: {screen_context or 'N/A'}

Output the JSON evaluation only."""

        try:
            r = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            msg = r.get("message", {}) or {}
            text = (msg.get("content") or "").strip()
            # strip markdown code block if present
            if "```" in text:
                for part in text.split("```"):
                    part = part.strip()
                    if part.startswith("json") or part.startswith("{"):
                        text = part.replace("json", "", 1).strip()
                        break
            data = json.loads(text)
            for key in ["technical_depth", "clarity", "originality", "understanding"]:
                if key not in data:
                    data[key] = 5
                else:
                    data[key] = max(0, min(10, int(data[key])))
            if "feedback_snippet" not in data:
                data["feedback_snippet"] = "No feedback generated."
            return data
        except Exception as e:
            return {
                "technical_depth": 5,
                "clarity": 5,
                "originality": 5,
                "understanding": 5,
                "feedback_snippet": f"Evaluation skipped ({e}).",
            }
