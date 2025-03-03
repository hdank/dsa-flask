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

    # Format each conversation entry as "Role: Content"
    formatted_history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in conversation_history 
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg
    ])

    # Retrieve context based on the conversation history and query
    context = (retrieve_relevant_documents(f"{formatted_history}\nUser: {query}", vector_store)
               if formatted_history else retrieve_relevant_documents(query, vector_store))

    # Format the retrieved context for inclusion in the prompt
    context_str = "\n".join([
        f"Document: {doc_info['content']}\nMetadata: {doc_info['metadata']}\nScore: {doc_info['score']}" 
        for doc_info in context
    ]) if context else "No relevant context found."

    augmented_prompt = (
        "You are a highly skilled technical assistant specializing in data structures and algorithms (DSA).\n\n"
        "### System Instructions:\n"
        "1. Answer the user's query primarily based on the context provided below. If the context lacks sufficient or relevant information, use your internal knowledge.\n"
        "2. Structure your response based on the type of query:\n\n"
        "   A. If the query is about understanding a data structure or algorithm concept:\n"
        "      - **CONCEPT**: Provide a clear, concise explanation of the concept\n"
        "      - **EXAMPLE**: Give a practical example to illustrate the concept\n"
        "      - **VISUALIZATION**: DO NOT PROVIDE DIRECT YOUTUBE LINKS. Instead, you MUST use the 'get_educational_video' function to get accurate video links.\n\n"
        "   B. If the query is about implementing or using a data structure or algorithm:\n"
        "      - **IMPLEMENTATION**: Provide clean, well-commented C/C++ code that implements the solution\n"
        "      - **EXPLANATION**: Explain how the code works, step-by-step\n"
        "      - **COMPLEXITY**: Analyze the time and space complexity of the solution\n\n"
        "3. Always format code blocks properly with syntax highlighting.\n"
        "4. CRITICAL INSTRUCTION: DO NOT INCLUDE ANY YOUTUBE LINKS OR URLS IN YOUR TEXT RESPONSE. The system will automatically append relevant educational video links after your response.\n\n"
        "### Retrieved Context:\n"
        f"{context_str}\n\n"
        "### Conversation History:\n"
        f"{formatted_history}\n\n"
        "### User Query:\n"
        f"{query}\n\n"
        "### Assistant Response:\n"
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



