"""Speech-to-Text helpers using faster-whisper for uploaded audio files."""
from dataclasses import dataclass
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    is_final: bool = True


class SpeechToTextService:
    """Transcribe uploaded audio files with faster-whisper."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _get_model(self):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed. pip install faster-whisper")
        if self._model is None:
            compute_type = "float16" if self.device == "cuda" else "int8"
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
        return self._model

    def transcribe_file(self, path: Path) -> list[TranscriptSegment]:
        model = self._get_model()
        segments, _ = model.transcribe(
            str(path),
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return [
            TranscriptSegment(text=segment.text.strip(), start=segment.start, end=segment.end, is_final=True)
            for segment in segments
            if segment.text.strip()
        ]

    def transcribe_text(self, path: Path) -> str:
        return " ".join(segment.text for segment in self.transcribe_file(path) if segment.text).strip()
