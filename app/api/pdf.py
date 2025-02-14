import time
from flask import request, Response, jsonify, json
from langchain_community.document_loaders import PDFPlumberLoader
from app.core.vector_store import store_documents, delete_document_by_id, retrieve_relevant_documents, get_vector_store
from app.core.config import PDF_FOLDER
import os, base64
from app.core.llm import stream_chat_response
from app.core.utils import manage_conversation, cleanup_old_conversations
from app.api.chat import rag_streaming_response
from app.core.utils import manage_conversation, cleanup_old_conversations, get_conversation_history, save_conversation


def ask_llama():
    print("Post /ask_llama called")
    query = request.form['query']
    conversation_id = request.form.get('conversation_id')
    
    # Manage conversation lifecycle
    conversation_id = manage_conversation(conversation_id)
    print(f"Using conversation ID: {conversation_id}")
    
    # Periodically cleanup old conversations
    cleanup_old_conversations()
    
    def generate_response():
        current_history = []
        try:
            # Get current conversation history
            conversation_data = get_conversation_history(conversation_id)
            current_history = conversation_data.get('history', [])
            # Get the system prompt from RAG function
            system_prompt = rag_streaming_response(query, current_history)
            
            # The prompt includes the system instructions, conversation history, and current query.
            prompt = f"### System:\n{system_prompt}\n\n"
            
            # Append previous conversation history (if any)
            for msg in current_history:
                if msg['role'] == 'user':
                    prompt += f"### User:\n{msg['content']}\n\n"
                else:
                    prompt += f"### Assistant:\n{msg['content']}\n\n"
            
            # Append the current query
            prompt += f"### User:\n{query}\n\n### Assistant:\n"
            
            # Update history with the new user query (store only the query)
            current_history.append({
                'role': 'user',
                'content': query,
                'created_at': time.time()
            })

            conversation_data['history'] = current_history

            save_conversation(conversation_id, conversation_data)
            
            messages_for_model = [{"role": "user", "content": prompt}]
            stream = stream_chat_response(messages_for_model, "llama")
            vector_store = get_vector_store()
            docs = retrieve_relevant_documents(query, vector_store)
            full_response = []
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='')
                full_response.append(content)

                response_chunk = {
                    'answer': content, 
                    'conversation_id': conversation_id,
                    'is_new_conversation': len(current_history) <= 1,
                    'docs': docs
                }
                response_string = json.dumps(response_chunk, ensure_ascii=False)
                # Encode to bytes for Server-Sent Events
                try:
                    yield f"data: {response_string}\n\n".encode('utf-8')
                except UnicodeEncodeError as e:
                    print(f"Encoding error: {e}")
                    error_response = {'error': f"UnicodeEncodeError: {str(e)}", 'conversation_id': conversation_id, 'is_new_conversation': len(current_history) <= 1}
                    yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode('utf-8')
            
            # Combine all chunks into the final assistant response
            complete_response = ''.join(full_response)
            current_history.append({
                'model': 'llama3.2',
                'role': 'assistant',
                'content': complete_response,
                'docs': docs,
                'created_at': time.time()
            })
            
            # Save the updated conversation history
            save_conversation(conversation_id, {
                'history': current_history,
                'last_activity': time.time()
            })
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            error_response = {
                'error': str(e), 
                'conversation_id': conversation_id,
                'is_new_conversation': len(current_history) <= 1
            }
            try:
                error_string = json.dumps(error_response, ensure_ascii=False)
                yield f"data: {error_string}\n\n".encode('utf-8')
            except UnicodeEncodeError as e:
                print(f"Encoding error in error handling: {e}")
                # If even the error message can't be encoded, provide a basic fallback
                yield f"data: {json.dumps({'error': 'An unexpected error occurred.'})}\n\n".encode('utf-8')
    
    return Response(
        generate_response(),
        content_type='text/event-stream; charset=utf-8'
    )

def ask_llama_vision():
    print("Post /ask_pdf called")
    query = request.form['query']
    image_file = request.files.get('image')
    conversation_id = request.form.get('conversation_id')
    
    # Handle image data
    image_base64 = None
    if image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Manage conversation lifecycle
    conversation_id = manage_conversation(conversation_id)
    print(f"Using conversation ID: {conversation_id}")
    
    # Periodically cleanup old conversations
    cleanup_old_conversations()
    
    def generate_response():
        current_history = []
        try:
            # Get current conversation history
            conversation_data = get_conversation_history(conversation_id)
            current_history = conversation_data.get('history', [])
            
            # Get the system prompt from RAG function
            system_prompt = rag_streaming_response(query, current_history)
            
            # The prompt includes the system instructions, conversation history, and current query.
            prompt = f"### System:\n{system_prompt}\n\n"
            
            # Append previous conversation history (if any)
            for msg in current_history:
                if msg['role'] == 'user':
                    prompt += f"### User:\n{msg['content']}\n\n"
                else:
                    prompt += f"### Assistant:\n{msg['content']}\n\n"
            
            # Append the current query
            prompt += f"### User:\n{query}\n\n### Assistant:\n"

            conversation_data['history'] = current_history
            
            # Update history with the new user query (store only the query)
            current_history.append({
                'model': 'llama3.2-vision',
                'role': 'user',
                'content': query,
                'images': image_base64,
                'created_at': time.time()
            })
            save_conversation(conversation_id, conversation_data)
            
            messages_for_model = [{"role": "user", "content": prompt,'images': [image_base64] if image_base64 else None}]
            stream = stream_chat_response(messages_for_model, "llama_vision")
            vector_store = get_vector_store()
            docs = retrieve_relevant_documents(query, vector_store)
            full_response = []
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='')
                full_response.append(content)

                response_chunk = {
                    'answer': content, 
                    'conversation_id': conversation_id,
                    'is_new_conversation': len(current_history) <= 1,
                    'docs': docs
                }
                response_string = json.dumps(response_chunk, ensure_ascii=False)
                # Encode to bytes for Server-Sent Events
                try:
                    yield f"data: {response_string}\n\n".encode('utf-8')
                except UnicodeEncodeError as e:
                    print(f"Encoding error: {e}")
                    error_response = {'error': f"UnicodeEncodeError: {str(e)}", 'conversation_id': conversation_id, 'is_new_conversation': len(current_history) <= 1}
                    yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n".encode('utf-8')
            
            # Combine all chunks into the final assistant response
            complete_response = ''.join(full_response)
            current_history.append({
                'role': 'assistant',
                'content': complete_response,
                'docs': docs,
                'created_at': time.time()
            })
            
            # Save the updated conversation history
            save_conversation(conversation_id, {
                'history': current_history,
                'last_activity': time.time()
            })
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            error_response = {
                'error': str(e), 
                'conversation_id': conversation_id,
                'is_new_conversation': len(current_history) <= 1
            }
            try:
                error_string = json.dumps(error_response, ensure_ascii=False)
                yield f"data: {error_string}\n\n".encode('utf-8')
            except UnicodeEncodeError as e:
                print(f"Encoding error in error handling: {e}")
                # If even the error message can't be encoded, provide a basic fallback
                yield f"data: {json.dumps({'error': 'An unexpected error occurred.'})}\n\n".encode('utf-8')
    
    return Response(
        generate_response(),
        content_type='text/event-stream; charset=utf-8'
    )

def pdf_post():
    file = request.files["file"]
    file_name = file.filename
    save_file = os.path.join(PDF_FOLDER, file_name)
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    document_id = store_documents(docs)

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "document_id": document_id  # Return the auto-generated IDs
    }
    return response

def delete_pdf():
    try:
        json_content = request.json
        file_name = json_content.get("file_name")+".pdf"
        document_id = json_content.get("document_id")

        # Delete the document and its chunks from vector store
        delete_document_by_id(document_id)

        # Delete the PDF file if it exists
        save_file = os.path.join(PDF_FOLDER, file_name)
        if os.path.exists(save_file):
            os.remove(save_file)
            print(f"Deleted PDF file: {save_file}")

        return {
            "status": 200,
            "message": f"Successfully deleted document {document_id} and file {file_name}"
        }
        
    except Exception as e:
        return {
            "status": 500,
            "error": f"Error deleting document: {str(e)}"
        }, 500