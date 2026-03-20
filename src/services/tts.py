import asyncio

try:
    import edge_tts
except ImportError:
    edge_tts = None


class TextToSpeechService:
    def __init__(self, voice: str = "en-US-JennyNeural"):
        self.voice = voice
        self._supported = edge_tts is not None

    def is_available(self) -> bool:
        return self._supported

    async def speak(self, text: str) -> None:
        if not text.strip() or not self._supported:
            return
        communicate = edge_tts.Communicate(text, self.voice)
        async for _ in communicate.stream():
            pass

    async def speak_non_blocking(self, text: str) -> asyncio.Task:
        return asyncio.create_task(self.speak(text))
