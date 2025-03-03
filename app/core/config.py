import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_FOLDER = os.path.join(BASE_DIR, "db")
PDF_FOLDER = os.path.join(BASE_DIR, "pdf")

# LLM Model
LLAMA_VISION = "llama3.2-vision"
LLAMA = "llama3.2"
QWEN_VISION = "qwen2-vl-2b-instruct"

# Embedding Model
EMBEDDING_MODEL = "fastembed"

# Text Splitter
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 80

# Conversation Management
CONVERSATION_TIMEOUT = 30 * 86400  # 30 days

# Prompt templates (optional, may also be placed in core/prompts.py)
RAG_PROMPT = """
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""

# OpenAI-compatible API settings
OPENAI_API_BASE_URL = "http://localhost:1234/v1"
OPENAI_MODEL = "llama-3.2-1b-instruct"
OPENAI_VISION_MODEL = "qwen2-vl-7b-instruct"
USE_OPENAI_COMPATIBLE_API = True  # Set to True to use OpenAI-compatible API instead of Ollama

# Settings object for API keys
class Settings:
    def __init__(self):
        self.YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "AIzaSyAUVzCue5nQpNORzXS4pgh2fmj3J9o3yko")

settings = Settings()