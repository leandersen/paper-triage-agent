"""Pure unit tests for paper_tools.py."""
from pathlib import Path

import pytest

from src.paper_tools import extract_text


# A paper we've already downloaded in earlier runs. Adjust if the cached file
# is somewhere else on your machine.
CACHED_PDF = Path("papers/2506.06962v3.pdf")


@pytest.fixture
def cached_pdf_path():
    """Skip the test if the cached PDF isn't present (e.g. on a fresh clone)."""
    if not CACHED_PDF.exists():
        pytest.skip(f"Cached PDF not found at {CACHED_PDF}; run the pipeline once first.")
    return str(CACHED_PDF)


def test_extract_text_returns_non_empty(cached_pdf_path):
    """Smoke test: extracted text should not be empty."""
    text = extract_text(cached_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 1000, f"expected substantial text, got {len(text)} chars"


def test_extract_text_respects_max_chars(cached_pdf_path):
    """The max_chars argument should be honored."""
    text = extract_text(cached_pdf_path, max_chars=500)
    assert len(text) <= 500


def test_extract_text_contains_arxiv_marker(cached_pdf_path):
    """Real arXiv PDFs include an arXiv identifier on the first page."""
    text = extract_text(cached_pdf_path)
    assert "arXiv" in text or "arxiv" in text.lower(), \
        "expected the extracted text to mention 'arXiv'"