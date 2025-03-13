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

def get_dynamic_k(query: str, base_k: int = 7, words_per_increment: int = 5, max_increment: int = 5) -> int:
    """
    Calculate a dynamic k value based on the number of words in the query.
    Every 'words_per_increment' words in the query increases k by 1,
    up to a maximum additional 'max_increment'.
    """
    word_count = len(query.split())
    # Calculate increment: one extra document per words_per_increment words, capped by max_increment
    increment = min(max_increment, word_count // words_per_increment)
    dynamic_k = base_k + increment
    print(f"Query word count: {word_count}, setting retrieval k to: {dynamic_k}")
    return dynamic_k
     
def retrieve_relevant_documents(query, vector_store, base_k: int = 3) -> list:
    # Add caching for similar queries
    import hashlib
    from functools import lru_cache
    
    # Enhanced exercise detection with more specific terms
    is_exercise_query = any(keyword in query.lower() for keyword in [
        "bài tập", "lab", "exercise", "practice", "thực hành", "ví dụ bài tập", "challenge",
        "triển khai", "implementation", "code", "viết chương trình", "thực hiện", "làm bài"
    ])
    
    # Improved algorithm-specific boosting with bilingual terms
    algorithm_specific_boosts = {
        "linear search": ["linear search", "tìm kiếm tuyến tính", "sequential search", 
                         "tìm kiếm tuần tự", "brute force search"],
        "binary search": ["binary search", "tìm kiếm nhị phân", "logarithmic search"],
        "bubble sort": ["bubble sort", "sắp xếp nổi bọt"],
        "selection sort": ["selection sort", "sắp xếp chọn"],
        # Add more algorithms as needed
    }
    
    # Check which algorithm is being queried
    detected_algorithm = None
    for algo, terms in algorithm_specific_boosts.items():
        if any(term in query.lower() for term in terms):
            detected_algorithm = algo
            break
            
    # Default boosting
    boosted_query = query
    adjusted_base_k = base_k
    filter_dict = None
    
    # Create a more specific query for exercises
    if is_exercise_query:
        exercise_terms = "bài tập exercise thực hành lab implementation code triển khai ví dụ challenge practice"
        
        if detected_algorithm:
            # Algorithm-specific boost with precise terms
            algorithm_terms = " ".join(algorithm_specific_boosts.get(detected_algorithm, []))
            boosted_query = f"{query} {algorithm_terms} {exercise_terms}"
            # Increase base_k more for specific algorithm requests
            adjusted_base_k = base_k + 4
            
            # Add metadata filter to prioritize pages with exercise content
            try:
                # Find pages that might contain exercises for this topic
                prefilter_results = vector_store.similarity_search_with_score(
                    f"{detected_algorithm} exercise", k=2
                )
                if prefilter_results:
                    possible_pages = [doc.metadata.get('page') for doc, _ in prefilter_results if doc.metadata.get('page')]
                    if possible_pages:
                        # Create a filter to prefer these pages but not exclude others
                        filter_dict = {"page": {"$in": possible_pages}}
                        print(f"Added page filter for exercises: {filter_dict}")
            except Exception as e:
                print(f"Error creating filter: {e}")
                filter_dict = None
        else:
            # General exercise boost
            boosted_query = f"{query} {exercise_terms}"
            adjusted_base_k = base_k + 2
            
        # Use MMR for diverse results that include both theory and exercises
        try:
            mmr_results = vector_store.max_marginal_relevance_search(
                boosted_query, 
                k=adjusted_base_k,
                fetch_k=adjusted_base_k*3,  # Fetch more for better diversity
                lambda_mult=0.7  # Increase diversity (0.5-0.7 is a good range)
            )
            
            # Convert to the expected format with scores
            docs = []
            for doc in mmr_results:
                # Approximate score for MMR results (not actually used for ranking)
                synthetic_score = 0.5
                
                # Boost score for exercise content
                content_lower = doc.page_content.lower()
                if any(kw in content_lower for kw in ["exercise", "bài tập", "ví dụ", "task", "thực hành", "bài lab", "lab", "homework"]):
                    synthetic_score = 0.2  # Lower score means higher relevance
                
                # Extra boost for very likely exercise content
                if any(marker in content_lower for marker in 
                      ["viết chương trình", "implement", "code the following", 
                       "write a program", "write code", "thực hiện thuật toán"]):
                    synthetic_score = 0.1  # Even better score
                
                doc_dict = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': synthetic_score
                }
                docs.append(doc_dict)
            
            # Sort by score (ascending is better)
            docs.sort(key=lambda x: x['score'])
            
            return docs
            
        except Exception as e:
            print(f"Error in MMR search: {e}, falling back to standard search")
            # Fall through to standard search
    
    # Use standard cached retrieval as fallback
    @lru_cache(maxsize=128)
    def _cached_retrieval(query_hash, base_k, boosted=False, filter_str=None):
        search_query = boosted_query if boosted else query
        dynamic_k = get_dynamic_k(search_query, base_k=base_k)
        
        try:
            # Try with filter if available
            if filter_dict and filter_str:
                docs_and_scores = vector_store.similarity_search_with_score(
                    search_query, k=dynamic_k, filter=eval(filter_str)
                )
            else:
                docs_and_scores = vector_store.similarity_search_with_score(search_query, k=dynamic_k)
        except Exception as e:
            print(f"Error with filter, falling back to standard search: {e}")
            docs_and_scores = vector_store.similarity_search_with_score(search_query, k=dynamic_k)
        
        if not docs_and_scores:
            print("No relevant context found.")
            return []

        docs = []
        for doc, score in docs_and_scores:
            # For exercise queries, boost documents that contain exercise-related keywords
            if is_exercise_query:
                content_lower = doc.page_content.lower()
                
                # More sophisticated exercise content detection with weighted boosting
                exercise_boost = 1.0
                if any(kw in content_lower for kw in ["exercise", "bài tập", "task", "practice", "thực hành"]):
                    exercise_boost *= 0.7  # Strong boost
                
                if any(kw in content_lower for kw in ["implement", "code", "viết", "triển khai"]):
                    exercise_boost *= 0.8  # Medium boost
                    
                if detected_algorithm and any(term in content_lower for term in algorithm_specific_boosts.get(detected_algorithm, [])):
                    exercise_boost *= 0.75  # Algorithm-specific boost
                
                # Apply the combined boost
                if exercise_boost < 1.0:
                    score = score * exercise_boost

            doc_dict = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            }
            docs.append(doc_dict)

        # Sort by score (ascending as lower is better)
        docs.sort(key=lambda x: x['score'])
        
        return docs
    
    # Create a hash of the query and parameters for caching
    filter_str = str(filter_dict) if filter_dict else None
    query_hash = hashlib.md5((boosted_query if is_exercise_query else query).encode()).hexdigest()
    
    # First try with boosting and filtering if it's an exercise query
    if is_exercise_query:
        results = _cached_retrieval(query_hash, adjusted_base_k, True, filter_str)
        if results:
            return results
    
    # Fallback to standard retrieval if needed
    return _cached_retrieval(query_hash, base_k)
    
def store_documents(docs):
    """
    Enhanced document storage with improved chunking strategy for exercises
    """
    # Generate a unique document ID and check if it already exists
    import time
    document_id = f"{str(uuid.uuid4())}_{int(time.time())}"  # Add timestamp for extra uniqueness
    
    # Use a smaller chunk size with more overlap for exercise content
    exercise_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # Smaller chunk size
        chunk_overlap=150,       # Reduced overlap to reduce memory usage
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Process chunks with metadata enrichment
    chunks = []
    for doc in docs:
        # Check if this document might contain exercises
        content_lower = doc.page_content.lower()
        is_exercise_doc = any(keyword in content_lower for keyword in 
            ["bài tập", "practice", "exercise", "lab", "thực hành", "triển khai"])
            
        # Choose appropriate splitter
        splitter = exercise_text_splitter if is_exercise_doc else text_splitter
        doc_chunks = splitter.split_documents([doc])
        
        # Enrich metadata for each chunk
        for i, chunk in enumerate(doc_chunks):
            # Preserve original metadata
            chunk.metadata.update(doc.metadata)
            
            # Add additional metadata with unique timestamp
            chunk.metadata["document_id"] = document_id
            chunk.metadata["chunk_id"] = f"{document_id}_chunk_{i}"
            
            # Tag content type
            if is_exercise_doc:
                chunk.metadata["content_type"] = "exercise"
                # Add more specific tagging for algorithm types
                if "linear search" in content_lower or "tìm kiếm tuyến tính" in content_lower:
                    chunk.metadata["algorithm"] = "linear_search"
                elif "binary search" in content_lower or "tìm kiếm nhị phân" in content_lower:
                    chunk.metadata["algorithm"] = "binary_search"
                # Add more algorithm tags as needed
                
            chunks.append(chunk)
    
    try:
        # Create unique IDs for each chunk with safety check
        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        
        # Check for duplicates in the generated IDs (should never happen with our new ID format)
        if len(chunk_ids) != len(set(chunk_ids)):
            print("Warning: Duplicate IDs detected in new chunks")
            # Make them unique by adding counter
            unique_ids = []
            for i, chunk_id in enumerate(chunk_ids):
                unique_ids.append(f"{chunk_id}_{i}")
            chunk_ids = unique_ids
        
        # Check if any chunks with these IDs already exist in ChromaDB
        try:
            vector_store = get_vector_store()
            existing_ids = vector_store._collection.get(ids=chunk_ids, include=[])['ids']
            
            if existing_ids:
                print(f"Found {len(existing_ids)} existing chunks with the same IDs. Generating new IDs.")
                # Generate completely new IDs to avoid conflict
                new_document_id = f"{str(uuid.uuid4())}_{int(time.time())}_new"
                for i, chunk in enumerate(chunks):
                    chunk.metadata["document_id"] = new_document_id
                    chunk.metadata["chunk_id"] = f"{new_document_id}_chunk_{i}"
                
                chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
                
        except Exception as e:
            # If checking fails (e.g., empty DB), proceed with original IDs
            print(f"Error checking existing IDs: {e}, proceeding with generated IDs")
            
        # Store the chunks in ChromaDB
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=DB_FOLDER,
            ids=chunk_ids,
        )
        vector_store.persist()
        
        return document_id
        
    except chromadb.errors.DuplicateIDError as e:
        print(f"Error: Duplicate IDs detected when adding to ChromaDB: {e}")
        
        # Alternative approach: Update existing entries instead of adding new ones
        # Get existing vector store
        vector_store = get_vector_store()
        
        # For each chunk, try to update if exists or add if new
        for chunk, chunk_id in zip(chunks, chunk_ids):
            try:
                # Try to delete the existing chunk first
                try:
                    vector_store._collection.delete(ids=[chunk_id])
                except Exception:
                    pass  # Ignore if it doesn't exist
                
                # Now add the chunk with the same ID
                vector_store.add_documents(
                    documents=[chunk],
                    ids=[chunk_id]
                )
            except Exception as inner_e:
                print(f"Error updating chunk {chunk_id}: {inner_e}")
        
        vector_store.persist()
        return document_id

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