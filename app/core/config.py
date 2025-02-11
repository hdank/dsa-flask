import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_FOLDER = os.path.join(BASE_DIR, "db")
PDF_FOLDER = os.path.join(BASE_DIR, "pdf")

# LLM Model
LLAMA_VISION = "llama3.2-vision"
LLAMA = "llama3.2"

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