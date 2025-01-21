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

chat_history = []

folder_path = "db"

cached_llm = Ollama(model="llama3.2")

vision_llm = Ollama(model="llava:7b")

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

def rag_streaming_response(query):
    context = retrieve_context(query)

    augmented_prompt = f"You are a technical assistant good at searching documents and i will give you the Context, answer the question from Input. If the Context do not have an answer from the provided information say so. Context: {context}. Input: {query}"

    return augmented_prompt

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
         'images': [image_base64]  # Pass the base64-encoded image
        }
    ]
    
    # Assuming chat is a function that processes the messages
    model_name = "llava:7b"  # Ensure this is a string
    res = chat(model=model_name, messages=messages)
    print(res['message']['content'])
    return jsonify({"response": res['message']['content']})


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    # Generate a conversation ID
    conversation_id = str(uuid.uuid4())

    def generate_response():
        # Send the conversation ID as the first message
        initial_response = {
            "conversation_id": conversation_id,
            "type": "conversation_id"  # Add type to distinguish from answer chunks
        }
        yield f"data: {json.dumps(initial_response)}\n\n"

        # Then send the regular streaming response
        stream = chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': f'{rag_streaming_response(query)}'}],
            stream=True,
        )
        for chunk in stream:
            response_answer = {
                "answer": chunk['message']['content'],
                "type": "content"  # Add type to distinguish from conversation_id
            }
            print(chunk['message']['content'], end='', flush=True)
            yield f"data: {json.dumps(response_answer)}\n\n"

    return Response(generate_response(), content_type='text/event-stream')

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
