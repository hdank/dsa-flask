from langchain.prompts import PromptTemplate
from app.core.config import RAG_PROMPT

raw_prompt = PromptTemplate.from_template(RAG_PROMPT)

def get_rag_prompt_template():
    return raw_prompt