"""Orchestrates screen OCR, STT, LLM questions, TTS, and evaluation for the live interview."""
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from .screen_ocr import ScreenOCRService, ScreenContent
from .stt import SpeechToTextService, TranscriptSegment
from .tts import TextToSpeechService
from .llm import LLMService
from .evaluator import EvaluatorService, FeedbackReport


@dataclass
class InterviewState:
    """Mutable state for one interview session."""
    started_at: float = 0.0
    current_question: str = ""
    questions_asked: list[str] = field(default_factory=list)
    last_screen_content: str = ""
    last_transcript: str = ""
    report: FeedbackReport = field(default_factory=lambda: FeedbackReport(
        overall_score=0.0,
        technical_depth=0.0,
        clarity=0.0,
        originality=0.0,
        understanding=0.0,
        summary="",
    ))
    phase: str = "presenting"  # presenting | asking | listening | ended
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)


class InterviewOrchestrator:
    """
    Runs the full interview flow:
    - Capture screen periodically (OCR)
    - Transcribe speech (STT)
    - After min_presentation_seconds, generate questions from context
    - Speak questions (TTS), wait for answer (STT), evaluate (LLM), repeat until max_questions
    - Produce final FeedbackReport
    """

    def __init__(
        self,
        screen_ocr: ScreenOCRService,
        stt: SpeechToTextService,
        tts: TextToSpeechService,
        llm: LLMService,
        evaluator: EvaluatorService,
        *,
        max_questions: int = 5,
        min_presentation_seconds: float = 30.0,
        answer_wait_seconds: float = 15.0,
    ):
        self.screen_ocr = screen_ocr
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.evaluator = evaluator
        self.max_questions = max_questions
        self.min_presentation_seconds = min_presentation_seconds
        self.answer_wait_seconds = answer_wait_seconds
        self.state = InterviewState()
        self._on_state_change: Callable[[InterviewState], Awaitable[None]] | None = None
        self._ocr_task: asyncio.Task | None = None
        self._stt_task: asyncio.Task | None = None
        self._refresh_task: asyncio.Task | None = None
        self._running = False

    def set_state_callback(self, callback: Callable[[InterviewState], Awaitable[None]]):
        self._on_state_change = callback

    async def _notify_state(self):
        if self._on_state_change:
            await self._on_state_change(self.state)

    async def _run_ocr_loop(self):
        """Keep updating last_screen_content from OCR."""
        async for content in self.screen_ocr.stream_captures():
            if not self._running:
                break
            self.state.last_screen_content = content.text
            await self._notify_state()

    async def _transcript_refresh_loop(self):
        """Every 1.5s refresh last_transcript and push state so UI updates frequently."""
        while self._running:
            await asyncio.sleep(1.5)
            if not self._running:
                break
            if self.stt._recording:
                self.state.last_transcript = self.stt.get_full_transcript()
                await self._notify_state()

    async def _on_transcript(
        self,
        segments: list[TranscriptSegment],
        error: str | None = None,
    ):
        """Called when new transcript segments arrive (or on error). Always refresh transcript for UI."""
        if error:
            self.state.last_transcript = f"[Microphone error: {error}]"
        else:
            self.state.transcript_segments.extend(segments)
            self.state.last_transcript = self.stt.get_full_transcript()
        await self._notify_state()

    async def run(self) -> FeedbackReport:
        """
        Run full interview: present → ask questions → evaluate → report.
        Blocks until interview is done.
        """
        self.state.started_at = time.time()
        self.state.phase = "presenting"
        self._running = True
        self.screen_ocr._running = True

        # Start OCR and STT in background
        self._ocr_task = asyncio.create_task(self._run_ocr_loop())
        self._stt_task = asyncio.create_task(
            self.stt.run_transcription_loop(self._on_transcript)
        )
        # Push transcript to UI every 1.5s so user sees updates between 5s transcription chunks
        self._refresh_task = asyncio.create_task(self._transcript_refresh_loop())
        await self._notify_state()

        # Minimum presentation time
        await asyncio.sleep(self.min_presentation_seconds)
        if not self._running:
            return self._finalize()

        questions_done = 0
        while self._running and questions_done < self.max_questions:
            # Generate next question
            self.state.phase = "asking"
            question = await asyncio.to_thread(
                self.llm.generate_question,
                self.state.last_screen_content,
                self.state.last_transcript,
                self.state.questions_asked,
                "followup" if self.state.questions_asked else "context",
            )
            self.state.current_question = question
            self.state.questions_asked.append(question)
            await self._notify_state()

            # Speak question
            if self.tts.is_available():
                await self.tts.speak(question)
            await asyncio.sleep(0.5)

            # Listen for answer
            self.state.phase = "listening"
            answer_start_len = len(self.stt.get_full_transcript())
            await asyncio.sleep(self.answer_wait_seconds)
            full = self.stt.get_full_transcript()
            answer = full[answer_start_len:] if len(full) > answer_start_len else full
            if not answer.strip():
                answer = "(No verbal answer captured.)"

            await self._notify_state()

            # Evaluate
            eval_data = await asyncio.to_thread(
                self.llm.evaluate_response,
                question,
                answer,
                self.state.last_screen_content,
            )
            self.evaluator.add_evaluation(
                self.state.report,
                question,
                answer,
                eval_data,
            )
            questions_done += 1
            self.state.phase = "presenting" if questions_done < self.max_questions else "ended"
            await self._notify_state()

        return self._finalize()

    def _finalize(self) -> FeedbackReport:
        self._running = False
        self.screen_ocr.stop()
        self.stt.stop_listening()
        if self._ocr_task and not self._ocr_task.done():
            self._ocr_task.cancel()
        if self._stt_task and not self._stt_task.done():
            self._stt_task.cancel()
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
        self.evaluator.finalize_report(self.state.report)
        self.state.phase = "ended"
        return self.state.report

    def stop(self):
        """External stop (e.g. user clicked End)."""
        self._running = False
