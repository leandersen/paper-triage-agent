"""Test: simple agent SDK call."""
import anyio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
from dotenv import load_dotenv

load_dotenv()

async def main():
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful research assistant.",
        max_turns = 1,
    )
    async for message in query(
        prompt="In one sentence, what makes a research paper 'good'?",
        options = options,
    ):
        # gives multiple message types, but we only care about the final
        # assistant text for this test.
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")

if __name__ == "__main__":
    anyio.run(main)
