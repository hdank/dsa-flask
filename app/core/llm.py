from ollama import chat
from app.core.config import LLAMA, LLAMA_VISION

def stream_llm_response(augmented_prompt, model):
    if model == "llama_vision":
        return chat(
                model=LLAMA_VISION,
                messages=[{'role': 'user', 'content': f'{augmented_prompt}'}],
                stream=True,
            )
    else:
        return chat(
                model=LLAMA_VISION,
                messages=[{'role': 'user', 'content': f'{augmented_prompt}'}],
                stream=True,
            )

def stream_chat_response(messages, model):
    # Ensure messages is a list of dictionaries
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a list of dictionaries")
    if model == "llama_vision":
        return chat(
                model=LLAMA_VISION,
                messages=messages,
                stream=True,
            )
    else:
        return chat(
                model=LLAMA_VISION,
                messages=messages,
                stream=True,
            )
