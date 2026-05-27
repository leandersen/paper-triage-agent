"""Eval: does the synthesizer actually cite the sources it was given?"""
import pytest

from src.graph_app import synthesize_node


# A tiny, deterministic fixture: two papers with obvious cite-able IDs.
FAKE_NOTES = [
    {
        "arxiv_id": "1234.56789",
        "title": "On the Reduction of Hallucination via Retrieval",
        "methodology": "Authors evaluate RAG on TriviaQA.",
        "key_findings": ["RAG reduces hallucinations by 35%."],
        "limitations": ["Tested only on TriviaQA."],
        "relevance_score": 9,
        "relevance_reasoning": "Directly addresses the question.",
    },
    {
        "arxiv_id": "9876.54321",
        "title": "Hierarchical Retrieval for Vision-Language Models",
        "methodology": "ARA framework with active triggering.",
        "key_findings": ["10-point F1 improvement on POPE."],
        "limitations": ["Vision-language only, not pure LLM."],
        "relevance_score": 6,
        "relevance_reasoning": "Adjacent but not directly LLM-focused.",
    },
]


@pytest.fixture(scope="module")
def synthesized_report():
    """Run synthesize_node once and reuse the output across all tests."""
    state = {
        "research_question": "How can retrieval augmentation reduce hallucinations in LLMs?",
        "search_results": [],
        "selected_ids": [],
        "paper_notes": FAKE_NOTES,
        "final_report": "",
    }
    result = synthesize_node(state)
    return result["final_report"]


@pytest.mark.slow
def test_report_is_non_trivial(synthesized_report):
    """A real report should be at least a few hundred characters."""
    assert len(synthesized_report) > 300, \
        f"report suspiciously short: {len(synthesized_report)} chars"


@pytest.mark.slow
def test_report_cites_first_paper(synthesized_report):
    """The synthesizer should mention the arxiv_id we gave it."""
    assert "1234.56789" in synthesized_report, \
        "report didn't cite the first paper's arxiv_id"


@pytest.mark.slow
def test_report_cites_second_paper(synthesized_report):
    """Both papers should appear, not just the first."""
    assert "9876.54321" in synthesized_report, \
        "report didn't cite the second paper's arxiv_id"


@pytest.mark.slow
def test_report_doesnt_fabricate_extra_citations(synthesized_report):
    """The report shouldn't invent arxiv IDs we never gave it.

    We check this by looking for common 'arxiv:' prefixes followed by digits
    that AREN'T one of our two known IDs.
    """
    import re
    # Find anything that looks like an arxiv id reference
    candidates = re.findall(r"\b\d{4}\.\d{4,5}\b", synthesized_report)
    allowed = {"1234.56789", "9876.54321"}
    fabricated = set(candidates) - allowed
    assert not fabricated, \
        f"report invented arxiv IDs that weren't in input: {fabricated}"