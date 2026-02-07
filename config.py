"""Application configuration."""
import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        """App settings from env or defaults."""

        ollama_base_url: str = "http://localhost:11434"
        ollama_model: str = "llama3.2"
        whisper_model_size: str = "base"
        whisper_device: str = "auto"
        audio_sample_rate: int = 16000
        audio_chunk_seconds: float = 5.0
        audio_input_device: int | None = None  # sounddevice input device index; None = default
        screen_capture_interval_sec: float = 8.0
        ocr_languages: list[str] = ["en"]
        max_questions: int = 5
        min_presentation_seconds: float = 30.0
        project_root: Path = Path(__file__).resolve().parent
        reports_dir: Path = Path(__file__).resolve().parent / "reports"
        temp_dir: Path = Path(__file__).resolve().parent / "tmp"

        class Config:
            env_prefix = "INTERVIEW_"
            env_file = ".env"

except ImportError:
    # Fallback without pydantic-settings: use os.environ
    class Settings:
        def __init__(self):
            p = Path(__file__).resolve().parent
            self.ollama_base_url = os.environ.get("INTERVIEW_OLLAMA_BASE_URL", "http://localhost:11434")
            self.ollama_model = os.environ.get("INTERVIEW_OLLAMA_MODEL", "llama3.2")
            self.whisper_model_size = os.environ.get("INTERVIEW_WHISPER_MODEL_SIZE", "base")
            self.whisper_device = os.environ.get("INTERVIEW_WHISPER_DEVICE", "auto")
            self.audio_sample_rate = int(os.environ.get("INTERVIEW_AUDIO_SAMPLE_RATE", "16000"))
            self.audio_chunk_seconds = float(os.environ.get("INTERVIEW_AUDIO_CHUNK_SECONDS", "5.0"))
            _dev = os.environ.get("INTERVIEW_AUDIO_INPUT_DEVICE")
            self.audio_input_device = int(_dev) if _dev not in (None, "") else None
            self.screen_capture_interval_sec = float(os.environ.get("INTERVIEW_SCREEN_CAPTURE_INTERVAL_SEC", "8.0"))
            self.ocr_languages = ["en"]
            self.max_questions = int(os.environ.get("INTERVIEW_MAX_QUESTIONS", "5"))
            self.min_presentation_seconds = float(os.environ.get("INTERVIEW_MIN_PRESENTATION_SECONDS", "30.0"))
            self.project_root = p
            self.reports_dir = p / "reports"
            self.temp_dir = p / "tmp"


def get_settings() -> Settings:
    return Settings()
