"""Constants and flags shared by more than one example script. Per-chapter
constants (sample questions, chunk sizes, PDF paths) stay in their own
script.
"""

import argparse
from pathlib import Path

PDF_URL = "https://arxiv.org/pdf/1709.00666.pdf"
LOG_DIR = Path("logs")
DEFAULT_ANSWER_MODEL = "gpt-4o"


def add_reindex_argument(parser: argparse.ArgumentParser) -> None:
    """``--reindex``: indexes persist in each script's Kuzu database and
    are reused, unless this asks to build them again.
    """
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the index (or graph) even if its Kuzu database already holds one.",
    )
