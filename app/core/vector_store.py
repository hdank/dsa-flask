from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import DB_FOLDER, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, is_separator_regex=False
)


def get_vector_store():
     return Chroma(persist_directory=DB_FOLDER, embedding_function=embedding)
     
def retrieve_relevant_documents(query, vector_store, k=3):
    docs = vector_store.similarity_search(query, k=k)
    if not docs:
        print("No relevant context found.")
    return "\n".join([doc.page_content for doc in docs])
    
def store_documents(docs):
    chunks = text_splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding, 
        persist_directory=DB_FOLDER
    )
    vector_store.persist()
    
    # Get the ID of the first chunk which we'll use as document ID
    collection = vector_store._collection
    ids = collection.get()['ids']
    document_id = ids[0] if ids else None  # Get first ID
    return  document_id

def delete_document_by_id(document_id):
     vector_store = Chroma(
            persist_directory=DB_FOLDER,
            embedding_function=embedding
        )
     vector_store._collection.delete(ids=[document_id])
     vector_store.persist()