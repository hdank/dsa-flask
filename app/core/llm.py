from langchain_community.llms import Ollama
from ollama import chat
from app.core.config import OLLAMA_MODEL, LLAMA_MODEL

cached_llm = Ollama(model=OLLAMA_MODEL)

def stream_llm_response(augmented_prompt):
    return chat(
            model=LLAMA_MODEL,
            messages=[{'role': 'user', 'content': f'{augmented_prompt}'}],
            stream=True,
        )

def stream_chat_response(messages):
    # Ensure messages is a list of dictionaries
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a list of dictionaries")

    return chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True
    )