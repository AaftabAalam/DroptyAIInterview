"""Speech-to-Text using faster-whisper (local, free)."""
import asyncio
import queue
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


def list_audio_input_devices() -> list[dict]:
    """Return list of input devices for microphone selection. Each dict has index, name, channels, sample_rate."""
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            result.append({
                "index": i,
                "name": dev.get("name", "Unknown"),
                "channels": dev["max_input_channels"],
                "default_sample_rate": dev.get("default_samplerate", 44100),
            })
    return result

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


@dataclass
class TranscriptSegment:
    """One segment of transcribed speech."""
    text: str
    start: float
    end: float
    is_final: bool = True


class SpeechToTextService:
    """Records microphone and transcribes with faster-whisper in chunks."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        sample_rate: int = 16000,
        chunk_seconds: float = 5.0,
        input_device: int | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.input_device = input_device  # sounddevice device index; None = default
        self._model = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._stream = None
        self._recording = False
        self._transcript_buffer: deque[TranscriptSegment] = deque(maxlen=200)

    def _get_model(self):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed. pip install faster-whisper")
        if self._model is None:
            # Use int8 on CPU and when device is "auto" (e.g. Mac); float16 only on CUDA
            use_float16 = self.device == "cuda"
            compute_type = "float16" if use_float16 else "int8"
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
        return self._model

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        self._audio_queue.put(indata.copy())

    def _transcribe_chunk(self, audio_data: np.ndarray) -> list[TranscriptSegment]:
        """Transcribe a chunk of audio; returns list of segments."""
        if audio_data.size == 0:
            return []
        model = self._get_model()
        # faster_whisper expects float32 in [-1, 1]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32768.0
        segments, info = model.transcribe(
            audio_data,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        result = []
        for s in segments:
            result.append(
                TranscriptSegment(
                    text=s.text.strip(),
                    start=s.start,
                    end=s.end,
                    is_final=True,
                )
            )
        return result

    def start_listening(self):
        """Start recording from microphone into internal queue."""
        if self._recording:
            return
        self._recording = True
        self._audio_queue = queue.Queue()
        block_size = int(self.sample_rate * 0.1)  # 100 ms blocks
        kwargs = dict(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            blocksize=block_size,
            callback=self._audio_callback,
        )
        if self.input_device is not None:
            kwargs["device"] = self.input_device
        self._stream = sd.InputStream(**kwargs)
        self._stream.start()

    def stop_listening(self):
        """Stop recording."""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_accumulated_audio(self) -> np.ndarray | None:
        """Drain queue and return concatenated audio (float32) or None if empty."""
        chunks = []
        try:
            while True:
                chunks.append(self._audio_queue.get_nowait())
        except queue.Empty:
            pass
        if not chunks:
            return None
        data = np.concatenate(chunks, axis=0)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / 32768.0
        return data.flatten()

    def transcribe_accumulated(self) -> list[TranscriptSegment]:
        """Take accumulated audio, transcribe it, return segments."""
        audio = self.get_accumulated_audio()
        min_samples = int(self.sample_rate * 0.3)  # at least 0.3 sec
        if audio is None or len(audio) < min_samples:
            return []
        segments = self._transcribe_chunk(audio)
        for s in segments:
            self._transcript_buffer.append(s)
        return segments

    def get_full_transcript(self) -> str:
        """Return full transcript so far as a single string."""
        return " ".join(s.text for s in self._transcript_buffer if s.text).strip()

    def get_recent_transcript(self, last_n: int = 20) -> str:
        """Return text from last N segments."""
        segments = list(self._transcript_buffer)[-last_n:]
        return " ".join(s.text for s in segments if s.text).strip()

    async def run_transcription_loop(self, callback):
        """
        Run in background: every chunk_seconds, transcribe accumulated audio
        and call callback(segments). Callback is always called so UI can
        show current full transcript even when this chunk had no new segments.
        """
        try:
            self.start_listening()
        except Exception as e:
            if callback:
                await callback([], error=str(e))
            return
        try:
            while self._recording:
                await asyncio.sleep(self.chunk_seconds)
                if not self._recording:
                    break
                try:
                    segments = await asyncio.to_thread(self.transcribe_accumulated)
                except Exception as e:
                    if callback:
                        await callback([], error=str(e))
                    continue
                if callback:
                    await callback(segments)
        finally:
            self.stop_listening()
