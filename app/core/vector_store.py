import uuid
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import DB_FOLDER, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
import chromadb
from chromadb.utils import embedding_functions

embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, is_separator_regex=False
)


def get_vector_store():
     return Chroma(persist_directory=DB_FOLDER, embedding_function=embedding)
     
def retrieve_relevant_documents(query, vector_store, k=3):
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    if not docs_and_scores:
        print("No relevant context found.")
        return []

    docs = []
    for doc, score in docs_and_scores:
        doc_dict = {
            'content': doc.page_content,
            'metadata': doc.metadata,
            'score': score
        }
        docs.append(doc_dict)

    return docs
    
def store_documents(docs):
    # Generate a unique document ID
    document_id = str(uuid.uuid4())
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(docs)
    
    # Create unique IDs for each chunk that include the document ID
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_FOLDER,
        ids=chunk_ids,
    )
    vector_store.persist()
    
    return document_id  # Return the generated document ID

def delete_document_by_id(document_id):
    vector_store = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embedding
    )
    
    # Delete all chunks associated with this document ID
    vector_store._collection.delete(
        where={"document_id": document_id}
    )
    vector_store.persist()