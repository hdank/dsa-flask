from flask import request, Response, json
from app.core.llm import stream_llm_response
from app.core.vector_store import retrieve_relevant_documents, get_vector_store
from app.core.utils import manage_conversation, cleanup_old_conversations
from app.core.prompts import get_rag_prompt_template

def rag_streaming_response(query, conversation_history):
    vector_store = get_vector_store()

    # Ensure conversation_history is a list
    if not isinstance(conversation_history, list):
        print(f"Warning: conversation_history is not a list: {conversation_history}")
        conversation_history = []  # Default to an empty list
    
    # Ensure every element in conversation_history is a dictionary
    formatted_history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in conversation_history if isinstance(msg, dict) and 'role' in msg and 'content' in msg
    ])

    context = retrieve_relevant_documents(f"{formatted_history}\nUser: {query}", vector_store) if formatted_history else retrieve_relevant_documents(query, vector_store)

    context_str = "\n".join([
        f"Document: {doc_info['content']}\nMetadata: {doc_info['metadata']}\nScore: {doc_info['score']}" 
        for doc_info in context
    ]) if context else "No relevant context found."

    augmented_prompt = (
        f"You are a technical assistant good at searching documents. I will give you the Context and you can answer the question from Input. "
        f"If the Context does not have an answer from the provided information, say so. "
        f"If the question is not relevant to the Context, you just answer as usual.\n\n"
        f"Context: {context_str}. Input: {query}"
    )

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