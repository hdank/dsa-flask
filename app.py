import os
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

app = Flask(__name__)

# Ensure directories exist
os.makedirs("pdf", exist_ok=True)
os.makedirs("db", exist_ok=True)

# Initialize global variables
folder_path = "db"
cached_llm = Ollama(model="llama3.2")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=80, 
    length_function=len, 
    is_separator_regex=False
)

# Improved prompt template
RAW_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following pieces of context to answer the question as accurately and concisely as possible. 
If the context does not contain enough information to answer the question, say "I don't have enough information to answer that based on the provided document."

Context:
{context}

Question: {input}

Answer:""")

@app.route("/ai", methods=["POST"])
def ai_post():
    """Simple AI endpoint without RAG"""
    try:
        json_content = request.json
        query = json_content.get("query")
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        response = cached_llm.invoke(query)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    """RAG endpoint for PDF querying"""
    try:
        json_content = request.json
        query = json_content.get("query")
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Load vector store
        vector_store = Chroma(
            persist_directory=folder_path, 
            embedding_function=embedding
        )
        
        # Configure retriever
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,  # Reduced number of retrieved documents
                "score_threshold": 0.3,  # Slightly increased threshold
            }
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(
            cached_llm, 
            RAW_PROMPT
        )
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Invoke chain
        result = retrieval_chain.invoke({"input": query})
        
        # Prepare sources
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content
            } for doc in result.get("context", [])
        ]
        
        return jsonify({
            "answer": result.get("answer", "No answer generated"),
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pdf", methods=["POST"])
def pdf_post():
    """Endpoint for uploading and indexing PDF"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        # Save file
        file_name = file.filename
        save_file = os.path.join("pdf", file_name)
        file.save(save_file)
        
        # Load and process PDF
        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        
        # Split into chunks
        chunks = text_splitter.split_documents(docs)
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding, 
            persist_directory=folder_path
        )
        vector_store.persist()
        
        return jsonify({
            "status": "Successfully Uploaded",
            "filename": file_name,
            "doc_len": len(docs),
            "chunks": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_app():
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    start_app()