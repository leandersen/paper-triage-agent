"""Claude + tool use loop, no framework"""
import json
from anthropic import Anthropic
from dotenv import load_dotenv

from arxiv_tool import search_arxiv

load_dotenv()
client = Anthropic()

# feed info about tool to Claude
TOOLS = [
    {
        "name":"search_arxiv",
        "description":(
            "Search the arXiv preprint repository for academic papers. "
            "Use this whenever the user asks about recent research, specific "
            "papers, or topics where you need up-to-date scientific literature."
        ),
        "input_schema": {
            "type":"object",
            "properties": {
                "query": {
                    "type":"string",
                    "description": "The search query, e.g. 'retrieval augmented generation'.",
                },
                "max_results": {
                    "type":"integer",
                    "description":"Number of papers to return (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }
]

def run_once(user_message:str):
    """Send one user message and print Claude's response (no loop yet)."""
    response = client.messages.create(
        model = 'claude-sonnet-4-5',
        max_tokens = 1024,
        tools = TOOLS,
        messages=[{'role':'user','content':user_message}]
    )
    print(f"stop_reason: {response.stop_reason}")
    print(f'content blocks: {len(response.content)}')
    for i, block in enumerate(response.content):
        print(f"\n--- Block {i} (type: {block.type}) ---")
        if block.type == 'text':
            print(block.text)
        elif block.type == 'tool_use':
            print(f"Tool: {block.name}")
            print(f"Input: {json.dumps(block.input, indent = 2)}")
            print(f"Tool use id: {block.id}")

def run_agent(user_message: str, max_turns: int = 10) -> str:
    """Run the agent loop until Claude is done or we hit max_turns.
    Returns Claude's final text response.
    """
    messages = [{'role':'user','content':user_message}]

    for turn in range(max_turns):
        print(f"\n=== Turn {turn+1} ===")
        response = client.messages.create(
            model='claude-sonnet-4-5',
            max_tokens = 2048,
            tools = TOOLS,
            messages = messages,
        )
        print(f"stop_reason: {response.stop_reason}")

        # Append Claude's response to the conversation, exactly as we received it
        messages.append({"role":"assistant", "content":response.content})

        # if claude is done, return the final text.
        if response.stop_reason == 'end_turn':
            final_text = ''.join(
                block.text for block in response.content if block.type == 'text'
            )
            return final_text
        
        # if claude request tool - run and send results back
        if response.stop_reason == 'tool_use':
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"    > Claude wants to call: {block.name}({block.input})")

                # dispatch by tool name.
                if block.name =='search_arxiv':
                    result = search_arxiv(**block.input)
                    result_str = json.dumps(result, indent = 2)
                
                else: result_str = f"Error: unknown tool {block.name}" 

                tool_results.append({
                    "type":"tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            # The "user" turn that responds to tool calls is fabricated by us.
            messages.append({"role":"user", "content":tool_results})

    return "Hit max turns without completing."

if __name__ == "__main__":
    answer = run_agent(
        "Find 3 recent papers about retrieval augmented generation, " \
        "then tell me which one looks most relevant to improving factual " \
        "accuracy in LLMs and why."
    )
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(answer)