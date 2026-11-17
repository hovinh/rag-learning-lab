"""Constants shared by more than one example script. Per-chapter constants
(sample questions, chunk sizes, PDF/DB paths) stay in their own script.
"""

from pathlib import Path

PDF_URL = "https://arxiv.org/pdf/1709.00666.pdf"
LOG_DIR = Path("logs")
DEFAULT_ANSWER_MODEL = "gpt-4o"
