"""Utilities for reading uploaded resume files into plain text."""
from pathlib import Path


class ResumeParseError(RuntimeError):
    """Raised when an uploaded resume file cannot be parsed."""


class ResumeParserService:
    """Read supported resume file types into text."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ResumeParseError("Unsupported file type. Please upload PDF, DOCX, TXT, or MD.")
        if suffix in {".txt", ".md"}:
            return self._read_text(path)
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".docx":
            return self._read_docx(path)
        raise ResumeParseError("Unsupported file type.")

    def _read_text(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise ResumeParseError("The uploaded text file is empty.")
        return text

    def _read_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ResumeParseError("PDF parsing dependency is missing. Install pypdf.") from exc

        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
        if not text:
            raise ResumeParseError("Could not extract readable text from the PDF resume.")
        return text

    def _read_docx(self, path: Path) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise ResumeParseError("DOCX parsing dependency is missing. Install python-docx.") from exc

        document = Document(str(path))
        text = "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()).strip()
        if not text:
            raise ResumeParseError("Could not extract readable text from the DOCX resume.")
        return text
