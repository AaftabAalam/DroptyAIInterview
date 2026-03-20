from dataclasses import dataclass
from pathlib import Path
import platform

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

    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _get_model(self):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed. pip install faster-whisper")
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        load_errors: list[str] = []
        for compute_type in self._candidate_compute_types():
            try:
                return WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
            except Exception as exc:
                load_errors.append(f"{compute_type}: {exc}")
        joined = "; ".join(load_errors) or "unknown loading error"
        raise RuntimeError(
            "Unable to load the Whisper model for transcription. "
            f"Tried compute types {', '.join(self._candidate_compute_types())}. Details: {joined}"
        )

    def _candidate_compute_types(self) -> list[str]:
        normalized_device = (self.device or "auto").lower()
        if normalized_device == "cuda":
            return ["float16", "int8_float16", "int8"]
        if normalized_device == "cpu":
            return ["int8", "float32"]
        if platform.system().lower() == "windows":
            return ["int8", "float32"]
        return ["int8", "float32"]

    def transcribe_file(self, path: Path) -> list[TranscriptSegment]:
        model = self._get_model()
        try:
            segments, _ = model.transcribe(
                str(path),
                language="en",
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
        except Exception as exc:
            raise RuntimeError(
                "Whisper could not decode the uploaded audio. "
                "On Windows, make sure the browser is opened on localhost/HTTPS and try recording again. "
                f"Original error: {exc}"
            ) from exc
        return [
            TranscriptSegment(text=segment.text.strip(), start=segment.start, end=segment.end, is_final=True)
            for segment in segments
            if segment.text.strip()
        ]

    def transcribe_text(self, path: Path) -> str:
        return " ".join(segment.text for segment in self.transcribe_file(path) if segment.text).strip()
