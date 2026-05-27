"""LangGraph orchestrator for the paper triage pipeline.

Nodes:
  search:     Lesson-2 hand-rolled agent finds candidate papers on arXiv
  triage:     simple Python heuristic picks the top-N to read
  read:       Lesson-3 Agent SDK reader produces structured notes per paper
  synthesize: a final Claude call composes a literature-review-style answer
"""
import json
from pathlib import Path
from typing import TypedDict

import anyio
from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from arxiv_tool import search_arxiv
from reader_agent import read_paper
import structlog
from logging_setup import configure_logging, log, new_run_id

load_dotenv()
configure_logging()
client = Anthropic()

# Shared state schema
class TriageState(TypedDict):
    research_question: str
    search_results: list[dict]
    selected_ids: list[str]
    paper_notes: list[dict]
    final_report: str

# Nodes
# Each function takes the full state and returns a partial update.
def search_node(state: TriageState) -> dict:
    """Find candidate papers on arXiv for the research question.

    Uses Claude to turn the user's natural-language question into a focused
    keyword query, since arXiv's API is keyword-match, not semantic.
    """
    log.info('search_node.start', question = state['research_question'][:80])

    # ask Claude for a focused search query.
    keyword_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                f"Convert this research question into a focused arXiv search query "
                f"of 3-6 keywords. Output ONLY the query, no quotes, no explanation.\n\n"
                f"Question: {state['research_question']}"
            ),
        }],
    )
    query = keyword_response.content[0].text.strip()
    log.info("search_node.keywords_extracted", query=query)

    # Step 2: run the actual search.
    results = search_arxiv(query, max_results=5)
    log.info("search_node.done", num_results=len(results), query=query)
    return {"search_results": results}

def triage_node(state: TriageState) -> dict:
    """Pick which papers to deeply read. For now: top 2 by arXiv's relevance order."""
    log.info("triage_node.start", candidates=len(state["search_results"]))
    ids = [p["id"] for p in state["search_results"][:2]]
    log.info("triage_node.done", selected=ids)
    return {"selected_ids": ids}

def read_node(state: TriageState) -> dict:
    """Read each selected paper using the Agent SDK reader. Sequential for now."""
    log.info("read_node.start", num_papers=len(state["selected_ids"]))
    notes = []
    for arxiv_id in state["selected_ids"]:
        log.info("read_node.paper_start", arxiv_id=arxiv_id)
        paper_notes = anyio.run(read_paper, arxiv_id, state["research_question"])
        notes.append({"arxiv_id": arxiv_id, **paper_notes})
        log.info(
            "read_node.paper_done",
            arxiv_id=arxiv_id,
            relevance_score=paper_notes.get("relevance_score"),
        )
    return {"paper_notes": notes}

def synthesize_node(state: TriageState) -> dict:
    """Combine all notes into a literature-review-style answer."""
    log.info("synthesize_node.start", num_notes=len(state["paper_notes"]))
    notes_summary = json.dumps(state['paper_notes'], indent=2)
    prompt = (
        f"Research question: {state['research_question']}\n\n"
        f"Here are structured notes from {len(state['paper_notes'])} papers I analyzed:\n\n"
        f"{notes_summary}\n\n"
        f"Write a brief literature-review-style answer to the research question. "
        f"Be honest: if the papers don't actually address the question well, say so. "
        f"Cite papers by their arxiv_id. Keep it under 400 words."
    )
    response = client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=1024,
        messages=[{'role':'user','content':prompt}],
    )
    report = response.content[0].text
    log.info("synthesize_node.done", report_chars=len(report))
    return {'final_report': report}

# build graph
def build_graph():
    builder = StateGraph(TriageState)
    builder.add_node("search", search_node)
    builder.add_node('triage', triage_node)
    builder.add_node('read', read_node)
    builder.add_node('synthesize', synthesize_node)

    builder.add_edge(START, 'search')
    builder.add_edge('search','triage')
    builder.add_edge('triage','read')
    builder.add_edge('read', 'synthesize')
    builder.add_edge('synthesize', END)

    return builder.compile()

if __name__ == '__main__':
    run_id = new_run_id()
    structlog.contextvars.bind_contextvars(run_id=run_id)
    log.info("pipeline.start", run_id=run_id)

    graph=build_graph()
    
    initial_state: TriageState = {
        "research_question":"How can retrieval augmentation reduce hallucinations in large language models?",
        "search_results": [],
        "selected_ids": [],
        "paper_notes": [],
        "final_report":"",
    }

    final_state=graph.invoke(initial_state)

    log.info("pipeline.done", report_chars=len(final_state["final_report"]))

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(final_state['final_report'])

    Path('runs').mkdir(exist_ok = True)
    out = Path("runs") / f"{run_id}.json"
    out.write_text(json.dumps(final_state, indent=2, default=str))
    log.info("pipeline.saved", path=str(out))