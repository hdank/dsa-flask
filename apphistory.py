import time
from flask import Flask, request, Response, stream_with_context, jsonify, json
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from ollama import chat
import os, logging, requests, click, shutil, uuid, base64

app = Flask(__name__)

# Global dictionary to store conversation histories and their last activity timestamp
conversation_histories = {}
conversation_metadata = {}

folder_path = "db"

cached_llm = Ollama(model="llava:7b")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)
def retrieve_context(query):
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    docs = vector_store.similarity_search(query, k=3)  # Retrieve top 3 documents
    print("Retrieved documents:", docs)
    if not docs:
        print("No relevant context found.")
    return "\n".join([doc.page_content for doc in docs])

def rag_streaming_response(query, conversation_history=None):
    if conversation_history:
        formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history])
        context = retrieve_context(f"{formatted_history}\nUser: {query}")
    else:
        context = retrieve_context(query)

    augmented_prompt = f"You are a technical assistant good at searching documents and i will give you the Context, answer the question from Input. If the Context do not have an answer from the provided information say so. Context: {context}. Input: {query}"

    return augmented_prompt

def manage_conversation(conversation_id=None):
    """
    Manages conversation lifecycle. Returns appropriate conversation_id.
    Creates new conversation only when needed.
    """
    current_time = time.time()
    
    # If conversation_id is provided, check if it exists and is still valid
    if conversation_id:
        if conversation_id in conversation_metadata:
            # Update last activity timestamp
            conversation_metadata[conversation_id]['last_activity'] = current_time
            return conversation_id
        else:
            # If the ID doesn't exist, we'll create a new one
            print(f"Conversation {conversation_id} not found, creating new conversation")
    
    # Create new conversation
    new_conversation_id = str(uuid.uuid4())
    conversation_histories[new_conversation_id] = []
    conversation_metadata[new_conversation_id] = {
        'created_at': current_time,
        'last_activity': current_time
    }
    return new_conversation_id

def cleanup_old_conversations():
    """
    Removes conversations that have been inactive for more than 24 hours
    """
    current_time = time.time()
    timeout = 24 * 3600  # 24 hours in seconds
    
    for conv_id in list(conversation_metadata.keys()):
        if current_time - conversation_metadata[conv_id]['last_activity'] > timeout:
            del conversation_histories[conv_id]
            del conversation_metadata[conv_id]
            print(f"Cleaned up inactive conversation: {conv_id}")

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    def generate_response():
        stream = chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': f'{rag_streaming_response(query)}'}],
            stream=True,
        )
        for chunk in stream:
            response_answer = {"answer": chunk['message']['content']}
            print(chunk['message']['content'], end='', flush=True)
            yield f"data: {response_answer}\n\n"  # Format for Server-Sent Events

    return Response(generate_response(), content_type='text/event-stream')

@app.route("/ask_image", methods=["POST"])
def askImage():
    print("POST /ask_image called")
    
    # Check if the image file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    
    # Check if the query is in the request
    if 'query' not in request.form:
        return jsonify({"error": "No query provided"}), 400
    
    query = request.form['query']

    # Read the image file and convert it to base64
    image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    messages = [
        {'role': 'user',
         'content': query,
         'images': [image_base64] 
        }
    ]
    
    model_name = "llava:7b" 
    res = chat(model=model_name, messages=messages)
    print(res['message']['content'])
    return jsonify({"response": res['message']['content']})



@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
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
            stream = chat(
                model='llava:7b',
                messages=messages_for_model,
                stream=True,
            )
            
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

# Add an endpoint to start a new conversation explicitly
@app.route("/new_conversation", methods=["POST"])
def start_new_conversation():
    conversation_id = manage_conversation(None)  # Force new conversation
    return jsonify({
        "status": "success",
        "conversation_id": conversation_id,
        "message": "New conversation started"
    })

# Add a new endpoint to retrieve conversation history
@app.route("/conversation_history/<conversation_id>", methods=["GET"])
def get_conversation_history(conversation_id):
    history = conversation_histories.get(conversation_id, [])
    return jsonify({
        "conversation_id": conversation_id,
        "history": history
    })

@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding, 
        persist_directory=folder_path
    )
    vector_store.persist()

    # Get the ID of the first chunk which we'll use as document ID
    collection = vector_store._collection
    ids = collection.get()['ids']
    document_id = ids[0] if ids else None  # Get first ID

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
        "document_id": document_id  # Return the Chroma-generated ID
    }
    return response

@app.route("/delete-pdf", methods=["DELETE"])
def delete_pdf():
    try:
        json_content = request.json
        file_name = json_content.get("file_name")
        document_id = json_content.get("document_id")

        # Initialize the vector store
        vector_store = Chroma(
            persist_directory=folder_path,
            embedding_function=embedding
        )

        # Delete the document and its chunks from vector store
        vector_store._collection.delete(ids=[document_id])
        vector_store.persist()

        # Delete the PDF file if it exists
        save_file = "pdf/" + file_name +".pdf"
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
    
    
def start_app():
    app.run(host="0.0.0.0", port=5000, debug=True)
    
if __name__ == "__main__":
    start_app()
