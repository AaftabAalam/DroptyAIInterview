"""FastAPI application for the Automated Interviewer demo."""
import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from src.services.screen_ocr import ScreenOCRService
from src.services.stt import SpeechToTextService
from src.services.tts import TextToSpeechService
from src.services.llm import LLMService
from src.services.evaluator import EvaluatorService
from src.services.orchestrator import InterviewOrchestrator, InterviewState

app = FastAPI(title="AI Automated Interviewer", version="1.0.0")
settings = get_settings()

# Static files (frontend)
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Globals: one orchestrator and report per run
_orchestrator: InterviewOrchestrator | None = None
_orchestrator_task: asyncio.Task | None = None
_ws_clients: list[WebSocket] = []
_last_report: dict[str, Any] | None = None


def _make_orchestrator() -> InterviewOrchestrator:
    screen_ocr = ScreenOCRService(
        capture_interval_sec=settings.screen_capture_interval_sec,
        languages=settings.ocr_languages,
    )
    stt = SpeechToTextService(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        sample_rate=settings.audio_sample_rate,
        chunk_seconds=settings.audio_chunk_seconds,
        input_device=getattr(settings, "audio_input_device", None),
    )
    tts = TextToSpeechService()
    llm = LLMService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    evaluator = EvaluatorService()
    return InterviewOrchestrator(
        screen_ocr=screen_ocr,
        stt=stt,
        tts=tts,
        llm=llm,
        evaluator=evaluator,
        max_questions=settings.max_questions,
        min_presentation_seconds=settings.min_presentation_seconds,
        answer_wait_seconds=15.0,
    )


def _state_to_dict(state: InterviewState) -> dict:
    return {
        "phase": state.phase,
        "current_question": state.current_question,
        "questions_asked": state.questions_asked,
        "last_screen_content": state.last_screen_content[:500] if state.last_screen_content else "",
        "last_transcript": state.last_transcript,
        "report": {
            "overall_score": state.report.overall_score,
            "technical_depth": state.report.technical_depth,
            "clarity": state.report.clarity,
            "originality": state.report.originality,
            "understanding": state.report.understanding,
            "summary": state.report.summary,
            "per_answer_len": len(state.report.per_answer),
        },
    }


async def _broadcast(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


@app.get("/api/audio/devices")
async def list_audio_devices():
    """List microphone input devices. Use 'index' in INTERVIEW_AUDIO_INPUT_DEVICE if default is wrong."""
    from src.services.stt import list_audio_input_devices
    return {"devices": list_audio_input_devices()}


@app.post("/api/audio/test")
async def test_microphone_and_transcribe():
    """
    Record 6 seconds from the default mic and run speech-to-text. Use this to verify
    the full pipeline (mic + faster-whisper) when the server is running.
    """
    import numpy as np
    import sounddevice as sd
    from src.services.stt import SpeechToTextService

    sample_rate = getattr(settings, "audio_sample_rate", 16000)
    device = getattr(settings, "audio_input_device", None)
    duration_sec = 6

    def record():
        rec = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
            device=device,
        )
        sd.wait()
        return rec

    try:
        rec = await asyncio.to_thread(record)
        audio = rec.flatten().astype(np.float32) / 32768.0
        peak = float(np.abs(rec).max())
    except Exception as e:
        return {"ok": False, "error": f"Recording failed: {e}", "transcript": ""}

    if peak < 100:
        return {
            "ok": False,
            "error": "No significant audio (peak level too low). Check mic or speak louder.",
            "transcript": "",
            "peak": peak,
        }

    stt = SpeechToTextService(
        model_size=getattr(settings, "whisper_model_size", "base"),
        device=getattr(settings, "whisper_device", "auto"),
        sample_rate=sample_rate,
        input_device=device,
    )
    try:
        segments = await asyncio.to_thread(stt._transcribe_chunk, audio)
        text = " ".join(s.text for s in segments if s.text).strip()
        return {
            "ok": True,
            "transcript": text or "(no speech recognized)",
            "peak": peak,
            "segments_count": len(segments),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "transcript": "", "peak": peak}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the demo UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse(
        "<h1>Automated Interviewer</h1><p>Place static/index.html here.</p>"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            try:
                _ = await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


@app.post("/interview/start")
async def interview_start():
    """Start the interview (presentation + questions). Runs in background."""
    global _orchestrator, _orchestrator_task, _last_report
    if _orchestrator_task and not _orchestrator_task.done():
        return {"status": "already_running", "message": "Interview already in progress."}
    _last_report = None
    _orchestrator = _make_orchestrator()

    async def state_callback(state: InterviewState):
        await _broadcast({"type": "state", "data": _state_to_dict(state)})

    _orchestrator.set_state_callback(state_callback)

    async def run_and_save_report():
        global _last_report
        report = await _orchestrator.run()
        settings = get_settings()
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        evaluator = EvaluatorService()
        path = settings.reports_dir / f"report_{int(time.time())}.html"
        evaluator.save_report(report, path)
        _last_report = {
            "overall_score": report.overall_score,
            "technical_depth": report.technical_depth,
            "clarity": report.clarity,
            "originality": report.originality,
            "understanding": report.understanding,
            "summary": report.summary,
            "report_url": f"/reports/{path.name}",
            "per_answer": [
                {
                    "question": a.question,
                    "answer": a.answer[:300],
                    "technical_depth": a.technical_depth,
                    "clarity": a.clarity,
                    "originality": a.originality,
                    "understanding": a.understanding,
                    "feedback_snippet": a.feedback_snippet,
                }
                for a in report.per_answer
            ],
        }
        await _broadcast({"type": "report", "data": _last_report})

    _orchestrator_task = asyncio.create_task(run_and_save_report())
    return {"status": "started", "message": "Interview started. Connect to /ws for live updates."}


@app.post("/interview/stop")
async def interview_stop():
    """Stop the running interview."""
    global _orchestrator
    if _orchestrator:
        _orchestrator.stop()
    return {"status": "stopping", "message": "Stop requested."}


@app.get("/interview/status")
async def interview_status():
    """Current state (for polling if not using WebSocket)."""
    out = {"state": None, "running": False, "report": _last_report}
    if _orchestrator and _orchestrator.state:
        out["state"] = _state_to_dict(_orchestrator.state)
        out["running"] = _orchestrator._running
    return out


@app.get("/interview/report")
async def interview_report():
    """Last generated report (after interview ends)."""
    if _last_report is None:
        return {"report": None}
    return {"report": _last_report}


@app.get("/reports/{filename}")
async def get_report_file(filename: str):
    """Serve a generated report HTML file."""
    path = get_settings().reports_dir / filename
    if not path.is_file():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(path)
