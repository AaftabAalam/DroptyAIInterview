"""Screen capture and OCR to extract text from the presenter's screen."""
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import mss
import numpy as np
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None


@dataclass
class ScreenContent:
    """Extracted content from one screen capture."""
    text: str
    timestamp: float
    raw_lines: list[str] = field(default_factory=list)


class ScreenOCRService:
    """Captures screen and runs OCR to extract text (UI, code, slides)."""

    def __init__(
        self,
        capture_interval_sec: float = 8.0,
        languages: list[str] | None = None,
    ):
        self.capture_interval = capture_interval_sec
        self.languages = languages or ["en"]
        self._reader = None
        self._running = False

    def _get_reader(self):
        if easyocr is None:
            raise RuntimeError("easyocr is not installed. pip install easyocr")
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
        return self._reader

    def _capture_screen(self) -> np.ndarray:
        """Grab current screen as RGB numpy array."""
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # all monitors combined, or use [1] for primary
            screenshot = sct.grab(monitor)
            img = Image.frombytes(
                "RGB",
                (screenshot.width, screenshot.height),
                screenshot.rgb,
            )
            return np.array(img)

    def _run_ocr(self, image: np.ndarray) -> list[str]:
        """Run OCR on image and return list of text lines."""
        reader = self._get_reader()
        results = reader.readtext(image, detail=0, paragraph=False)
        lines = [line.strip() for line in results if line.strip()]
        return lines

    def extract_from_screen(self) -> ScreenContent:
        """Capture current screen and return extracted text."""
        image = self._capture_screen()
        lines = self._run_ocr(image)
        text = "\n".join(lines) if lines else ""
        return ScreenContent(
            text=text,
            timestamp=time.time(),
            raw_lines=lines,
        )

    async def stream_captures(self) -> AsyncIterator[ScreenContent]:
        """Async generator: capture screen every interval and yield OCR result."""
        self._running = True
        try:
            while self._running:
                try:
                    content = await asyncio.to_thread(
                        self.extract_from_screen
                    )
                    yield content
                except Exception as e:
                    yield ScreenContent(
                        text="",
                        timestamp=time.time(),
                        raw_lines=[],
                    )
                await asyncio.sleep(self.capture_interval)
        finally:
            self._running = False

    def stop(self):
        self._running = False
