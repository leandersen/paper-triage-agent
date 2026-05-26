import anyio
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
)
from dotenv import load_dotenv

load_dotenv()

async def main():
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful code-reading assistant. Be concise.",
        allowed_tools = ['Read'],
        permission_mode="acceptEdits",
        cwd='.'
    )
    prompt = (
        "Read the file arxiv_tool.py in the current directory and tell me, "
        "in 2-3 sentences, what its search_arxiv function does."
    )
    async for message in query(prompt=prompt, options = options):
        print(f"\n--- {type(message).__name__} ---")

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Text: {block.text}")
                elif isinstance(block, ToolUseBlock):
                    print(f"Tool Call: {block.name}({block.input})")
        elif isinstance(message, UserMessage):
            # the SDK fabricates user messages containing tool results,
            # exactly like did by hand in lesson 2
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    #truncate for readability
                    content_str = str(block.content)[:200]
                    print(f"Tool Result: {content_str}...")
        elif isinstance(message, ResultMessage):
            print(f" Done. Cost: ${message.total_cost_usd:.4f}, "
                  f"Turns: {message.num_turns}, " 
                  f"Duration: {message.duration_ms}ms")
            
if __name__ == "__main__":
    anyio.run(main)
            