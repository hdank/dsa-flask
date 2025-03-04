from flask import request, Response, json
import time
import logging
from app.core.llm import stream_llm_response
from app.core.vector_store import retrieve_relevant_documents, get_vector_store
from app.core.utils import manage_conversation, get_conversation_history, save_conversation
from app.core.config import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import request, Response, json
import time
import logging
from app.core.llm import stream_llm_response
from app.core.vector_store import retrieve_relevant_documents, get_vector_store
from app.core.utils import manage_conversation, get_conversation_history, save_conversation
from app.core.config import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rag_streaming_response(query, conversation_history):
    """Process a query with RAG and prepare the response."""
    # Get vector store
    vector_store = get_vector_store()

    # Format conversation history
    formatted_history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in conversation_history 
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg
    ])

    # Retrieve context based on the query and conversation history
    context = retrieve_relevant_documents(
        f"{formatted_history}\nUser: {query}" if formatted_history else query, 
        vector_store
    )

    # Format retrieved context
    context_str = "\n".join([
        f"Tài liệu: {doc_info['content']}\nMetadata: {doc_info['metadata']}\nĐộ liên quan: {doc_info['score']}" 
        for doc_info in context
    ]) if context else "Không tìm thấy ngữ cảnh liên quan."

    # Format the prompt using the system prompt template from config.py
    formatted_prompt = SYSTEM_PROMPT.format(
        context=context_str,
        history=formatted_history,
        input=query
    )

    logger.info(f"Generated prompt with context length: {len(context_str)}")
    
    return formatted_prompt, context

def ai_post():
    """API endpoint for processing AI chat requests"""
    try:
        json_content = request.json
        if not json_content:
            return Response(json.dumps({"error": "Missing request body"}), status=400, content_type="application/json")
            
        query = json_content.get("query")
        conversation_id = json_content.get("conversation_id")
        
        if not query:
            return Response(json.dumps({"error": "Missing 'query' in request body"}), status=400, content_type="application/json")
        
        logger.info(f"Processing request for conversation: {conversation_id}")
        
        # Get or create a conversation
        conversation_id = manage_conversation(conversation_id)
        
        # Get conversation history
        conversation_data = get_conversation_history(conversation_id)
        if not conversation_data:
            logger.warning(f"Failed to get conversation data for {conversation_id}")
            conversation_data = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "name": "New Conversation",
                "messages": []
            }
        
        conversation_history = conversation_data.get("messages", [])
        
        # Add user message to history
        user_message = {
            "role": "user",
            "content": query,
            "created_at": time.time()
        }
        conversation_history.append(user_message)
        
        # Set name based on first query if it's a new conversation
        if conversation_data.get("name") == "New Conversation":
            conversation_data["name"] = f'Tiêu đề "{query[:30]}"' if len(query) > 0 else "New Conversation"
            
        # Save the conversation with the user message
        save_conversation(conversation_id, conversation_data)

        def generate_response():
            try:
                formatted_prompt, context = rag_streaming_response(query, conversation_history)
                stream = stream_llm_response(formatted_prompt)
                
                full_response = ""
                for chunk in stream:
                    content = chunk['message']['content']
                    full_response += content
                    response_answer = {"answer": content, "conversation_id": conversation_id}
                    yield f"data: {json.dumps(response_answer)}\n\n"
                
                # Save assistant response to conversation history
                assistant_message = {
                    "role": "assistant",
                    "model": "llama-3.2-1b-instruct",
                    "content": full_response,
                    "created_at": time.time(),
                    "docs": context if context else []
                }
                conversation_history.append(assistant_message)
                
                # Update conversation data
                conversation_data["messages"] = conversation_history
                conversation_data["last_activity"] = time.time()
                save_conversation(conversation_id, conversation_data)
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                error_response = {"error": str(e), "conversation_id": conversation_id}
                yield f"data: {json.dumps(error_response)}\n\n"

        return Response(generate_response(), content_type='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return Response(json.dumps({"error": str(e)}), status=500, content_type="application/json")



