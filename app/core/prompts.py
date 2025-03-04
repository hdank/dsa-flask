from langchain.prompts import PromptTemplate
from app.core.config import SYSTEM_PROMPT

# Use the consolidated prompt from config
rag_prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

def get_rag_prompt_template():
    return rag_prompt