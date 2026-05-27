"""Schema/contract test for the reader agent. Calls Claude (slow, ~30-60s, ~$0.25/run)."""
import anyio
import pytest

from reader_agent import read_paper


# We test against a paper we know exists. Costs API tokens per run.
# Tagged "slow" so devs can skip these in fast iteration loops.
ARXIV_ID = "2506.06962v3"
QUESTION = "How can retrieval augmentation reduce hallucinations in LLMs?"


@pytest.fixture(scope="module")
def reader_output():
    """Run the reader once for this module; reuse across tests.

    `scope="module"` means pytest calls this fixture ONCE per test file,
    not once per test. So our 5 assertions below share one $0.25 API call.
    """
    return anyio.run(read_paper, ARXIV_ID, QUESTION)


@pytest.mark.slow
def test_reader_returns_dict(reader_output):
    assert isinstance(reader_output, dict), "reader should return a dict"


@pytest.mark.slow
def test_reader_has_all_required_keys(reader_output):
    required = {
        "title",
        "methodology",
        "key_findings",
        "limitations",
        "relevance_score",
        "relevance_reasoning",
    }
    missing = required - set(reader_output.keys())
    assert not missing, f"missing required keys: {missing}"


@pytest.mark.slow
def test_reader_types_are_correct(reader_output):
    assert isinstance(reader_output["title"], str)
    assert isinstance(reader_output["methodology"], str)
    assert isinstance(reader_output["key_findings"], list)
    assert isinstance(reader_output["limitations"], list)
    assert isinstance(reader_output["relevance_score"], int)
    assert isinstance(reader_output["relevance_reasoning"], str)


@pytest.mark.slow
def test_reader_lists_are_non_empty(reader_output):
    """A paper should yield at least one finding and one limitation."""
    assert len(reader_output["key_findings"]) >= 1
    assert len(reader_output["limitations"]) >= 1


@pytest.mark.slow
def test_reader_relevance_score_in_range(reader_output):
    score = reader_output["relevance_score"]
    assert 1 <= score <= 10, f"relevance_score {score} out of range [1, 10]"