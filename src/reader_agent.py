"""The paper reader agent.

Uses the agent SDK with custom MCP tools (download_pdf, extract_text)
to fetch arXiv papers and produce structured notes.
"""

import json
from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from dotenv import load_dotenv

from paper_tools import(
    NOTES_DIR,
    download_pdf as _download_pdf,
    extract_text as _extract_text,
)

load_dotenv()

# MCP definitions
@tool(
    "download_pdf",
    "Download an arXiv paper PDF given its arXiv ID. "
    "Returns the local file path where the PDF was saved.",
    {"arxiv_id":str},
)
async def download_pdf_tool(args: dict[str, Any]) -> dict[str, Any]:
    try:
        path = _download_pdf(args['arxiv_id'])
        return {"content": [{"type":"text", "text":f"Downloaded to: {path}"}]}
    except Exception as e:
        return {
            "content": [{"type":"text","text":f"Error extracting: {e}"}],
            "isError": True,
        }
    
@tool(
    "extract_text",
    "Extract text from a downloaded PDF. Takes the path returned by "
    "download_pdf. Returns the paper text (truncated to ~40k chars).",
    {"pdf_path": str},
)
async def extract_text_tool(args: dict[str, Any]) -> dict[str, Any]:
    try:
        text = _extract_text(args["pdf_path"])
        return {"content": [{"type": "text", "text": text}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error extracting: {e}"}],
            "isError": True,
        }
    
@tool(
    "save_notes",
    "Save structured notes about a paper as JSON to notes/<arxiv_id>.json "
    "The notes argument should be a JSON string with keys: title, methodology, "
    "key_findings, limitations, relevance_score (1-10), relevance_reasoning.",
    {"arxiv_id":str, "notes_json": str},
)
async def save_notes_tool(args: dict[str, Any]) -> dict[str, Any]:
    try:
        # validate it's parseable json before writing
        parsed = json.loads(args['notes_json'])
        out_path = NOTES_DIR / f"{args['arxiv_id']}.json"
        out_path.write_text(json.dumps(parsed, indent = 2))
        return {'content': [{"type": "text", "text":f"Saved notes to {out_path}"}]}
    except json.JSONDecodeError as e:
        return {
            "content":[{"type":"text", "text": f"Invalid JSON: {e}"}],
            "isError": True
        }
    
# Register all three tools into one in-process MCP server
paper_server = create_sdk_mcp_server(
    name = "paper_tools",
    version = "1.0.0",
    tools = [download_pdf_tool, extract_text_tool, save_notes_tool],
)

# Agent
SYSTEM_PROMPT = """You are a research-paper analyst.

Given an arXiv paper ID and a research question, you will:
1. Download the paper using the download_pdf tool.
2. Extract its text using the extract_text tool.
3. Analyze the text and produce structured notes.
4. Save the notes using the save_notes tool. The notes_json must be valid JSON
   with these exact keys:
   - title: str
   - methodology: str (1-2 sentences)
   - key_findings: list of strings
   - limitations: list of strings
   - relevance_score: int 1-10, where 10 = directly answers the research question
   - relevance_reasoning: str (1-2 sentences explaining the score)

Be honest about relevance. If the paper is only tangentially related to the
research question, score it low and say so. Do not confabulate findings that
aren't in the text.
"""

async def read_paper(arxiv_id: str, research_question: str) -> dict[str, Any]:
    """
    Run the reader agent on one paper. Return parsed notes.
    """
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"paper_tools":paper_server},
        allowed_tools=[
            "mcp__paper_tools__download_pdf",
            "mcp__paper_tools__extract_text",
            "mcp__paper_tools__save_notes",
        ],
        permission_mode='acceptEdits',
        max_turns=10,
    )
    prompt = (
        f"Research question: {research_question}\n\n"
        f"Analyze arXiv paper {arxiv_id}. Download it, extract the text, "
        f"produce structured notes, and save them. Then summarize your findings."
    )

    print(f"\nReading {arxiv_id}...\n")

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"{block.text}\n")
                elif isinstance(block, ToolUseBlock):
                    print(f"Calling {block.name}({list(block.input.keys())})")
        elif isinstance(message, UserMessage):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    preview = str(block.content)[:120].replace("\n", " ")
                    print(f"    > {preview}...")
        elif isinstance(message, ResultMessage):
            print(
                f"\nDone. ${message.total_cost_usd:.4f}, "
                f"{message.num_turns} turns, {message.duration_ms}ms"
            )
    
    notes_path = Path('notes') / f"{arxiv_id}.json"
    if notes_path.exists():
        return json.loads(notes_path.read_text())
    return {'error': 'agent did not save notes'}

if __name__ == '__main__':
    notes = anyio.run(
        read_paper,
        "2506.06962v3",
        "How can retrieval augmentation improve factual accuracy in LLMs?",
    )
    print("\n" + "=" * 60)
    print("STRUCTURED NOTES:")
    print("="*60)
    print(json.dumps(notes, indent = 2))