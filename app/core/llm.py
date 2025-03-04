from ollama import chat
from app.core.config import  OPENAI_API_BASE_URL, OPENAI_MODEL, USE_OPENAI_COMPATIBLE_API
import logging

logger = logging.getLogger(__name__)

# Import the OpenAI client only if needed
if USE_OPENAI_COMPATIBLE_API:
    from app.core.openai_client import OpenAICompatibleClient
    openai_client = OpenAICompatibleClient(OPENAI_API_BASE_URL)

def stream_llm_response(augmented_prompt, model="llama"):
    """Stream LLM response using OpenAI-compatible API."""
    if USE_OPENAI_COMPATIBLE_API:
        logger.info(f"Using OpenAI-compatible API with model {OPENAI_MODEL}")
        messages = [{'role': 'user', 'content': f'{augmented_prompt}'}]
        return openai_client.chat_completion(
            messages=messages,
            model=OPENAI_MODEL,
            stream=True
        )

def stream_chat_response(messages, model=None):  # model param kept for backwards compatibility
    """Stream chat response using OpenAI-compatible API."""
    # Ensure messages is a list of dictionaries
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a list of dictionaries")
    
    if USE_OPENAI_COMPATIBLE_API:
        logger.info(f"Using OpenAI-compatible API with model {OPENAI_MODEL}")
        return openai_client.chat_completion(
            messages=messages,
            model=OPENAI_MODEL,
            stream=True
        )
