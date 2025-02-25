import logging, time, json
from flask import Response
from app.core.utils import manage_conversation, cleanup_old_conversations, get_conversation_history, save_conversation
from app.core.naming import generate_conversation_name
from app.api.chat import rag_streaming_response
from app.core.vector_store import get_vector_store, retrieve_relevant_documents
from app.core.llm import stream_chat_response

def _process_llama_request(form_data, image_base64=None):
    """
    Process Llama request with or without image.
    
    Args:
        form_data: The form data from the request
        image_base64: Optional base64-encoded image
        
    Returns:
        Response: Server-sent events stream
    """
    query = form_data['query']
    conversation_id = form_data.get('conversation_id')
    
    # Manage conversation lifecycle
    conversation_id = manage_conversation(conversation_id)
    logging.info(f"Using conversation ID: {conversation_id}")
    
    # Periodically cleanup old conversations
    cleanup_old_conversations()
    
    return Response(
        _generate_llama_response(query, conversation_id, image_base64),
        content_type='text/event-stream; charset=utf-8'
    )

def _generate_llama_response(query, conversation_id, image_base64=None):
    """
    Generate streaming response from Llama model.
    
    Args:
        query: User query text
        conversation_id: Conversation identifier
        image_base64: Optional base64-encoded image
        
    Yields:
        Encoded server-sent events
    """
    current_history = []
    model_type = "llama_vision" if image_base64 else "llama"
    model_name = "llama3.2-vision" if image_base64 else "llama3.2"
    
    try:
        # Get current conversation history
        conversation_data = get_conversation_history(conversation_id)
        current_history = conversation_data.get('history', [])

        # Generate conversation name
        conversation_name = generate_conversation_name(current_history, query)
        logging.info(f"Generated conversation name: {conversation_name}")

        # Prepare prompt with system instructions and history
        system_prompt = rag_streaming_response(query, current_history)
        prompt = _build_prompt(system_prompt, current_history, query)
        
        # Update history with the new user query
        user_message = {
            'model': model_name,
            'role': 'user',
            'content': query,
            'created_at': time.time(),
            'conversation_name': conversation_name
        }
        
        # Add image if provided
        if image_base64:
            user_message['images'] = image_base64
            
        current_history.append(user_message)
        conversation_data['history'] = current_history
        save_conversation(conversation_id, conversation_data)
        
        # Prepare message for model
        messages_for_model = [{
            "role": "user", 
            "content": prompt
        }]
        
        # Add image to message if provided
        if image_base64:
            messages_for_model[0]['images'] = [image_base64]
        
        # Get relevant documents
        vector_store = get_vector_store()
        docs = retrieve_relevant_documents(query, vector_store)
        
        # Stream response from model
        full_response = []
        stream = stream_chat_response(messages_for_model, model_type)
        
        for chunk in stream:
            content = chunk['message']['content']
            logging.info(content)
            full_response.append(content)
            
            response_chunk = {
                'answer': content, 
                'conversation_id': conversation_id,
                'is_new_conversation': len(current_history) <= 1,
                'docs': docs,
                'name': conversation_name
            }
            
            yield _encode_sse_message(response_chunk, conversation_id, len(current_history) <= 1)
        
        # Save complete response to conversation history
        complete_response = ''.join(full_response)
        current_history.append({
            'model': model_name,
            'role': 'assistant',
            'content': complete_response,
            'docs': docs,
            'created_at': time.time()
        })
        
        # Save updated conversation
        save_conversation(conversation_id, {
            'history': current_history,
            'last_activity': time.time()
        })
        
    except Exception as e:
        logging.info(f"Error in generate_response: {str(e)}")
        error_response = {
            'error': str(e), 
            'conversation_id': conversation_id,
            'is_new_conversation': len(current_history) <= 1
        }
        yield _encode_sse_message(error_response, conversation_id, len(current_history) <= 1)

def _build_prompt(system_prompt, history, query):
    """
    Build full prompt with system instructions, history, and query.
    
    Args:
        system_prompt: System instructions
        history: Conversation history
        query: Current user query
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"### System:\n{system_prompt}\n\n"
    
    # Append previous conversation history
    for msg in history:
        if msg['role'] == 'user':
            prompt += f"### User:\n{msg['content']}\n\n"
        else:
            prompt += f"### Assistant:\n{msg['content']}\n\n"
    
    # Append current query
    prompt += f"### User:\n{query}\n\n### Assistant:\n"
    
    return prompt

def _encode_sse_message(data, conversation_id, is_new_conversation):
    """
    Encode data as server-sent event.
    
    Args:
        data: Message data to encode
        conversation_id: Conversation identifier
        is_new_conversation: Whether this is a new conversation
        
    Returns:
        bytes: Encoded SSE message
    """
    try:
        data_string = json.dumps(data, ensure_ascii=False)
        return f"data: {data_string}\n\n".encode('utf-8')
    except UnicodeEncodeError as e:
        logging.info(f"Encoding error: {e}")
        error_data = {
            'error': f"UnicodeEncodeError: {str(e)}", 
            'conversation_id': conversation_id,
            'is_new_conversation': is_new_conversation
        }
        return f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')