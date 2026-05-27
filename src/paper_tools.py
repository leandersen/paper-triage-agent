"""Pure Python helpers for paper fetching and text extraction.
These will be wrapped as MCP tools in reader_agent.py.
Keeping in their own module allows for unit-testing directly.
"""

import urllib.request
from pathlib import Path
from pypdf import PdfReader
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# local folders
PAPERS_DIR = Path("papers")
NOTES_DIR = Path("notes")
PAPERS_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)

NETWORK_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, urllib.error.URLError)),
    reraise=True,
    before_sleep=lambda retry_state: print(
        f"NOTICE: PDF download failed (attempt {retry_state.attempt_number}), "
        f"retrying in {retry_state.next_action.sleep:.1f}s..."
    ),
)

@NETWORK_RETRY
def download_pdf(arxiv_id: str) -> str:
    """Download an arXiv paper PDF to papers/. Returns the local file path."""
    # Normalize: arXiv ids sometimes come with version suffixes
    clean_id = arxiv_id.strip()
    target = PAPERS_DIR / f"{clean_id}.pdf"
    if target.exists():
        return str(target)
    
    url = f"https://arxiv.org/pdf/{clean_id}"
    # arxiv blocks default python urllib UA - pretend to be browser
    req = urllib.request.Request(url, headers = {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        target.write_bytes(response.read())
    
    return str(target)

def extract_text(pdf_path:str, max_chars: int = 40_000) -> str:
    """Pull text out of a PDF. Truncates to max_chars to keep prompts sane."""
    reader = PdfReader(pdf_path)
    chunks = []
    total = 0
    for page in reader.pages:
        page_text = page.extract_text() or ""
        chunks.append(page_text)
        total += len(page_text)
        if total >= max_chars:
            break
    text = "\n\n".join(chunks)
    return text[:max_chars]

if __name__ == '__main__':
    # smoke test: download and extract a known paper.
    path = download_pdf("2506.06962v3")
    print(f"Downloaded to: {path}")

    text = extract_text(path)
    print(f"\nExtracted {len(text)} characters. First 500:\n")
    print(text[:500])

