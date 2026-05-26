import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

response = client.messages.create(
    model='claude-haiku-4-5',
    max_tokens=200,
    messages = [
        {'role':'user','content':'In one sentence, what is arXiv?'}
    ]
)

print(response.content[0].text)