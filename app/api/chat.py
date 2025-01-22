from flask import request, Response, json
from app.core.llm import stream_llm_response
from app.core.vector_store import retrieve_relevant_documents, get_vector_store
from app.core.utils import manage_conversation, cleanup_old_conversations, conversation_histories
from app.core.prompts import get_rag_prompt_template

def rag_streaming_response(query, conversation_history=None):
    vector_store = get_vector_store()
    if conversation_history:
        formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history])
        context = retrieve_relevant_documents(f"{formatted_history}\nUser: {query}", vector_store)
    else:
        context = retrieve_relevant_documents(query, vector_store)
        
    augmented_prompt = f"You are a technical assistant good at searching documents and i will give you the Context, answer the question from Input. If the Context do not have an answer from the provided information say so. Context: {context}. Input: {query}"
    return augmented_prompt


def ai_post():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    def generate_response():
         augmented_prompt = rag_streaming_response(query)
         stream = stream_llm_response(augmented_prompt)
         for chunk in stream:
            response_answer = {"answer": chunk['message']['content']}
            print(chunk['message']['content'], end='', flush=True)
            yield f"data: {response_answer}\n\n"  # Format for Server-Sent Events

    return Response(generate_response(), content_type='text/event-stream')