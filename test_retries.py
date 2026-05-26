"""Verify our retry logic actually fires on network failure."""
import urllib.error
from unittest.mock import patch

from arxiv_tool import search_arxiv


def main():
    # Simulate two timeouts followed by a real call.
    # On the third attempt, tenacity will let urlopen run for real and succeed.
    call_count = {"n": 0}

    real_urlopen = __import__("urllib.request").request.urlopen

    def flaky_urlopen(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise TimeoutError(f"Simulated timeout #{call_count['n']}")
        return real_urlopen(*args, **kwargs)

    with patch("urllib.request.urlopen", side_effect=flaky_urlopen):
        try:
            results = search_arxiv("retrieval augmentation", max_results=2)
            print(f"\n✅ Got {len(results)} results after {call_count['n']} attempts")
            print(f"   First paper: {results[0]['title'][:60]}")
        except Exception as e:
            print(f"\n❌ Failed even after retries: {e}")
            print(f"   Total attempts: {call_count['n']}")


if __name__ == "__main__":
    main()