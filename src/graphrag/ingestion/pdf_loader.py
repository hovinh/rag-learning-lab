from __future__ import annotations

from pathlib import Path

import pdfplumber
import requests


class PdfLoader:
    """Downloads a PDF and extracts its text (ch02: PDF ingestion)."""

    def download(self, url: str, dest: Path, *, timeout: float = 30.0) -> Path:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest

    def extract_text(self, path: Path) -> str:
        with pdfplumber.open(path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
