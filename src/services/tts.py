"""Text-to-Speech for speaking interview questions (edge-tts, free)."""
import asyncio
from pathlib import Path

try:
    import edge_tts
except ImportError:
    edge_tts = None


class TextToSpeechService:
    """Speak text using edge-tts (free, no API key for standard voices)."""

    def __init__(self, voice: str = "en-US-JennyNeural"):
        self.voice = voice
        self._supported = edge_tts is not None

    def is_available(self) -> bool:
        return self._supported

    async def speak(self, text: str) -> None:
        """Speak text and wait until done."""
        if not text.strip() or not self._supported:
            return
        communicate = edge_tts.Communicate(text, self.voice)
        # Consume the generator to play
        async for _ in communicate.stream():
            pass

    async def speak_non_blocking(self, text: str) -> asyncio.Task:
        """Start speaking in background; returns task so caller can await or cancel."""
        return asyncio.create_task(self.speak(text))

    async def save_to_file(self, text: str, path: Path) -> None:
        """Save TTS output to file (e.g. for playback in browser)."""
        if not text.strip() or not self._supported:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(path))
