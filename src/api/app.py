"""FastAPI application for the resume-based interview flow with browser-recorded answers."""
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from src.services.evaluator import EvaluatorService, FeedbackReport
from src.services.llm import LLMService
from src.services.resume_parser import ResumeParseError, ResumeParserService
from src.services.stt import SpeechToTextService

app = FastAPI(title="AI Resume Interviewer", version="3.0.0")
settings = get_settings()

static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_ws_clients: list[WebSocket] = []
_last_report: dict[str, Any] | None = None


@dataclass
class InterviewSession:
    phase: str = "upload"  # upload | ready | answering | ended
    uploaded_filename: str = ""
    upload_message: str = ""
    upload_stage: str = "idle"
    resume_text: str = ""
    resume_profile: dict[str, Any] = field(default_factory=dict)
    current_question: str = ""
    questions_asked: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    last_transcript: str = ""
    feedback_notes: list[str] = field(default_factory=list)
    report: FeedbackReport = field(
        default_factory=lambda: FeedbackReport(
            overall_score=0.0,
            communication=0.0,
            technical_relevance=0.0,
            confidence=0.0,
            clarity=0.0,
            overall_quality=0.0,
            summary="",
        )
    )


class AnswerRequest(BaseModel):
    answer: str


_session = InterviewSession()


def _make_llm() -> LLMService:
    return LLMService(base_url=settings.ollama_base_url, model=settings.ollama_model)


def _make_stt() -> SpeechToTextService:
    return SpeechToTextService(model_size=settings.whisper_model_size, device=settings.whisper_device)


def _make_report_payload(report: FeedbackReport, report_url: str | None = None) -> dict[str, Any]:
    return {
        "overall_score": report.overall_score,
        "communication": report.communication,
        "technical_relevance": report.technical_relevance,
        "confidence": report.confidence,
        "clarity": report.clarity,
        "overall_quality": report.overall_quality,
        "summary": report.summary,
        "report_url": report_url,
        "per_answer": [
            {
                "question": item.question,
                "answer": item.answer[:300],
                "communication": item.communication,
                "technical_relevance": item.technical_relevance,
                "confidence": item.confidence,
                "clarity": item.clarity,
                "overall_quality": item.overall_quality,
                "feedback_snippet": item.feedback_snippet,
            }
            for item in report.per_answer
        ],
    }


def _state_to_dict() -> dict[str, Any]:
    return {
        "phase": _session.phase,
        "is_ready": bool(_session.resume_text),
        "uploaded_filename": _session.uploaded_filename,
        "upload_message": _session.upload_message,
        "upload_stage": _session.upload_stage,
        "current_question": _session.current_question,
        "answers_count": len(_session.answers),
        "max_questions": settings.max_questions,
        "last_transcript": _session.last_transcript,
        "feedback_notes": _session.feedback_notes[-5:],
        "report": _make_report_payload(_session.report) if _session.phase == "ended" and _session.report.per_answer else None,
    }


async def _broadcast(data: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


async def _push_state() -> None:
    await _broadcast({"type": "state", "data": _state_to_dict()})


def _reset_session() -> None:
    global _session
    _session = InterviewSession()


def _save_report(report: FeedbackReport) -> str:
    evaluator = EvaluatorService()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.reports_dir / f"report_{int(time.time())}.html"
    evaluator.save_report(report, path)
    return f"/reports/{path.name}"


def _finalize_current_session() -> dict[str, Any] | None:
    global _last_report
    if not _session.report.per_answer:
        _session.phase = "ended"
        _session.current_question = ""
        return None
    evaluator = EvaluatorService()
    evaluator.finalize_report(_session.report)
    _session.phase = "ended"
    _session.current_question = ""
    report_url = _save_report(_session.report)
    _last_report = _make_report_payload(_session.report, report_url)
    return _last_report


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>AI Resume Interviewer</h1><p>Place static/index.html here.</p>")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        await websocket.send_json({"type": "state", "data": _state_to_dict()})
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


@app.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    global _last_report
    _last_report = None
    _reset_session()

    suffix = Path(file.filename or "").suffix.lower()
    safe_name = Path(file.filename or "resume").name
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = settings.temp_dir / f"resume_{int(time.time())}{suffix}"
    temp_path.write_bytes(await file.read())

    parser = ResumeParserService()
    llm = _make_llm()
    _session.upload_message = "Uploading file..."
    _session.upload_stage = "uploading"
    await _push_state()

    try:
        _session.upload_message = "Reading text from the uploaded resume..."
        _session.upload_stage = "parsing"
        await _push_state()
        resume_text = await asyncio.to_thread(parser.extract_text, temp_path)
    except ResumeParseError as exc:
        _session.phase = "upload"
        _session.upload_stage = "error"
        _session.upload_message = str(exc)
        await _push_state()
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)

    _session.upload_message = "Extracting skills, projects, and experience with llama3.2..."
    _session.upload_stage = "extracting"
    await _push_state()
    profile = await asyncio.to_thread(llm.extract_resume_profile, resume_text)

    _session.phase = "ready"
    _session.uploaded_filename = safe_name
    _session.upload_stage = "complete"
    _session.upload_message = "File uploaded successfully."
    _session.resume_text = resume_text
    _session.resume_profile = profile
    await _push_state()
    return {"status": "ok", "message": _session.upload_message}


@app.post("/interview/start")
async def interview_start():
    if not _session.resume_text:
        return JSONResponse(
            {"status": "error", "message": "Upload a resume before starting the interview."},
            status_code=400,
        )
    if _session.phase == "ended" and len(_session.answers) >= settings.max_questions:
        return JSONResponse(
            {"status": "error", "message": "Interview already completed. Upload a new resume to restart."},
            status_code=400,
        )
    if _session.current_question and _session.phase == "answering":
        return {"status": "already_running", "question": _session.current_question}

    llm = _make_llm()
    question = await asyncio.to_thread(
        llm.generate_resume_question,
        _session.resume_profile,
        _session.questions_asked,
        _session.answers,
        len(_session.questions_asked) + 1,
    )
    _session.current_question = question
    _session.questions_asked.append(question)
    _session.last_transcript = ""
    _session.feedback_notes = []
    _session.phase = "answering"
    await _push_state()
    return {"status": "started", "question": question}


@app.post("/interview/transcribe")
async def interview_transcribe(audio: UploadFile = File(...)):
    if not _session.current_question:
        return JSONResponse(
            {"status": "error", "message": "Start the interview before transcribing an answer."},
            status_code=400,
        )

    suffix = Path(audio.filename or "").suffix.lower() or ".webm"
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = settings.temp_dir / f"answer_{int(time.time() * 1000)}{suffix}"
    temp_audio_path.write_bytes(await audio.read())

    stt = _make_stt()
    try:
        transcript = await asyncio.to_thread(stt.transcribe_text, temp_audio_path)
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": f"Transcription failed: {exc}"},
            status_code=400,
        )

    _session.last_transcript = transcript
    await _push_state()
    return {"status": "ok", "transcript": transcript}


@app.post("/interview/answer")
async def interview_answer(payload: AnswerRequest):
    answer = payload.answer.strip()
    if not answer:
        return JSONResponse({"status": "error", "message": "Answer cannot be empty."}, status_code=400)
    if not _session.current_question:
        return JSONResponse(
            {"status": "error", "message": "Start the interview before sending an answer."},
            status_code=400,
        )

    llm = _make_llm()
    evaluator = EvaluatorService()
    question = _session.current_question
    evaluation = await asyncio.to_thread(
        llm.evaluate_interview_answer,
        question,
        answer,
        _session.resume_profile,
    )
    evaluator.add_evaluation(_session.report, question, answer, evaluation)
    _session.answers.append(answer)
    _session.last_transcript = answer
    _session.feedback_notes.append(str(evaluation.get("feedback_snippet", "")).strip())

    if len(_session.answers) >= settings.max_questions:
        report_payload = _finalize_current_session()
        await _push_state()
        if report_payload:
            await _broadcast({"type": "report", "data": report_payload})
        return {"status": "completed", "report": report_payload}

    next_question = await asyncio.to_thread(
        llm.generate_resume_question,
        _session.resume_profile,
        _session.questions_asked,
        _session.answers,
        len(_session.questions_asked) + 1,
    )
    _session.current_question = next_question
    _session.questions_asked.append(next_question)
    _session.last_transcript = ""
    _session.phase = "answering"
    await _push_state()
    return {
        "status": "next_question",
        "question": next_question,
        "feedback_snippet": evaluation.get("feedback_snippet", ""),
    }


@app.post("/interview/stop")
async def interview_stop(reason: str = Form(default="Interview ended manually.")):
    if reason:
        _session.feedback_notes.append(reason)
    report_payload = _finalize_current_session()
    await _push_state()
    if report_payload:
        await _broadcast({"type": "report", "data": report_payload})
    return {"status": "stopped", "report": report_payload}


@app.get("/interview/status")
async def interview_status():
    return {"state": _state_to_dict(), "running": _session.phase == "answering", "report": _last_report}


@app.get("/interview/report")
async def interview_report():
    return {"report": _last_report}


@app.get("/reports/{filename}")
async def get_report_file(filename: str):
    path = get_settings().reports_dir / filename
    if not path.is_file():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(path)
