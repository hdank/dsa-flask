from flask import request, Response, jsonify, json
from langchain_community.document_loaders import PDFPlumberLoader
from app.core.vector_store import store_documents, delete_document_by_id
from app.core.config import PDF_FOLDER
import os, base64
from app.core.llm import stream_chat_response
from app.core.utils import manage_conversation, cleanup_old_conversations, conversation_histories
from app.api.chat import rag_streaming_response


def ask_pdf_post():
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
        try:
            # Get current conversation history
            current_history = conversation_histories[conversation_id]
            
            # Get RAG context with conversation history
            context = rag_streaming_response(query, current_history)
            
            # Prepare the full message for the model
            user_message = f'{query}\nContext: {context}'
            
            # Update history with user message
            current_history.append({
                'role': 'user',
                'content': user_message
            })
            
            # Keep only last 10 messages
            conversation_histories[conversation_id] = current_history[-10:]
            
            # Prepare messages for the model including history
            messages_for_model = []
            for msg in current_history:
                messages_for_model.append({
                    'role': msg['role'],
                    'content': msg['content'],
                    'images': [image_base64] if msg['role'] == 'user' and image_base64 else None
                })

            # Stream response from LLM
            stream = stream_chat_response(messages_for_model)
            
            # Collect full response for history
            full_response = []
            for chunk in stream:
                content = chunk['message']['content']
                full_response.append(content)
                response_chunk = {
                    'answer': content, 
                    'conversation_id': conversation_id,
                    'is_new_conversation': len(current_history) <= 1
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"
            
            # Update history with assistant's complete response
            complete_response = ''.join(full_response)
            current_history.append({
                'role': 'assistant',
                'content': complete_response
            })
            
            # Keep only last 10 messages after adding assistant response
            conversation_histories[conversation_id] = current_history[-10:]
            
        except Exception as e:
            error_response = {
                'error': str(e), 
                'conversation_id': conversation_id,
                'is_new_conversation': len(current_history) <= 1
            }
            yield f"data: {json.dumps(error_response)}\n\n"

    return Response(generate_response(), content_type='text/event-stream')

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
        "document_id": document_id  # Return the Chroma-generated ID
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