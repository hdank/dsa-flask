import logging
import spacy
import random
from typing import List
from langdetect import detect

try:
    from underthesea import pos_tag
except ImportError:
    pos_tag = None

# Load English NLP model.
nlp = spacy.load('en_core_web_sm')

def detect_language(text: str) -> str:
    """
    Detect language code from a given text.
    Returns 'vi' for Vietnamese if detected, otherwise 'en'.
    """
    try:
        lang = detect(text)
        return "vi" if lang.startswith("vi") else "en"
    except Exception:
        return "en"

def extract_keywords_en(conversation_history: List[dict], query: str) -> List[str]:
    """Extract English keywords (nouns/proper nouns) using spaCy."""
    key_words = []
    # Process the query
    doc = nlp(query)
    key_words.extend([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']])
    # Process conversation history messages
    for msg in conversation_history:
        content = msg.get('content', '')
        doc = nlp(content)
        key_words.extend([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']])
    return list(set(key_words))

def extract_keywords_vi(conversation_history: List[dict], query: str) -> List[str]:
    """
    Extract Vietnamese keywords (nouns/proper nouns) using underthesea if available.
    If underthesea is missing, fall back to a simple whitespace split.
    """
    key_words = []
    if pos_tag:
        # Process the query
        for word, tag in pos_tag(query):
            if tag in ['N', 'Np']:
                key_words.append(word)
        # Process each conversation message
        for msg in conversation_history:
            content = msg.get('content', '')
            for word, tag in pos_tag(content):
                if tag in ['N', 'Np']:
                    key_words.append(word)
    else:
        # Fallback: simple split (less accurate)
        key_words = query.split()
        for msg in conversation_history:
            key_words.extend(msg.get('content', '').split())
    return list(set(key_words))

def generate_text_from_llm(prompt: str, model: str = "llama") -> str:
    """
    Synchronously generate text from LLM.
    This function wraps streaming API (e.g. stream_chat_response) and collects the response.
    """
    from app.core.llm import stream_chat_response  # Assuming this function exists in your project

    messages_for_model = [{"role": "user", "content": prompt}]
    generated_text = ""
    try:
        # Collect all streamed chunks into a single string.
        stream = stream_chat_response(messages_for_model, model)
        for chunk in stream:
            generated_text += chunk['message']['content']
        return generated_text.strip()
    except Exception as e:
        raise ValueError(f"Error generating text from LLM: {e}")

def generate_conversation_name(conversation_history: List[dict], query: str) -> str:
    """
    Generate a creative, professional conversation title based on the conversation context and query.
    The function auto-detects the language from the query and then builds a generative prompt for the LLM.
    """
    # Detect language from the query.
    language = detect_language(query)
    
    # Initialize conversation_name as None
    conversation_name = None
    
    # Prepare a generative prompt including the query and the extracted topics.
    try:
        if language == "vi":
            prompt = f"Tạo tiêu đề ngắn gọn (không quá 10 từ) cho cuộc trò chuyện về: \"{query}\". Trả về CHÍNH XÁC tiêu đề, không có giải thích, không có dấu ngoặc kép."
        else:
            prompt = f"Create a short title (no more than 10 words) for a conversation about: \"{query}\". Return ONLY the title, no explanation, no quotes."
                
        # Call LLM with retry
        max_retries = 2
        for i in range(max_retries):
            try:
                conversation_name = generate_text_from_llm(prompt)
                # Verify response quality
                if conversation_name and len(conversation_name) > 5 and len(conversation_name) < 100:
                    # Clean up formatting
                    conversation_name = conversation_name.strip('"\'').strip()
                    break
            except Exception as inner_e:
                logging.warning(f"Retry {i+1}/{max_retries} for LLM title generation: {str(inner_e)}")
                time.sleep(1)  # Brief delay before retry
                
    except Exception as e:
        logging.error(f"Error using LLM for conversation name: {str(e)}")
    
    # Create topic-based titles if LLM fails
    if not conversation_name or conversation_name == "":
        # Check for common data structure/algorithm terms
        topic_mapping = {
            # Basic Data Structures
            'stack': 'Cấu trúc dữ liệu Stack' if language == 'vi' else 'Stack Data Structure',
            'queue': 'Cấu trúc dữ liệu Queue' if language == 'vi' else 'Queue Data Structure',
            'linked list': 'Cấu trúc dữ liệu Linked List' if language == 'vi' else 'Linked List Data Structure',
            'array': 'Mảng' if language == 'vi' else 'Array Data Structure',
            'hash table': 'Bảng băm' if language == 'vi' else 'Hash Table Data Structure',
            'hash map': 'Bảng băm' if language == 'vi' else 'Hash Map Data Structure',
            
            # Trees
            'binary tree': 'Cấu trúc dữ liệu Binary Tree' if language == 'vi' else 'Binary Tree Data Structure',
            'binary search tree': 'Cây nhị phân tìm kiếm' if language == 'vi' else 'Binary Search Tree',
            'avl tree': 'Cây AVL' if language == 'vi' else 'AVL Tree',
            'b-tree': 'Cây B' if language == 'vi' else 'B-Tree',
            'heap': 'Cấu trúc Heap' if language == 'vi' else 'Heap Data Structure',
            
            # Graphs
            'graph': 'Đồ thị' if language == 'vi' else 'Graph Data Structure',
            'directed graph': 'Đồ thị có hướng' if language == 'vi' else 'Directed Graph',
            'undirected graph': 'Đồ thị vô hướng' if language == 'vi' else 'Undirected Graph',
            
            # Sorting Algorithms
            'quick sort': 'Thuật toán Quick Sort' if language == 'vi' else 'Quick Sort Algorithm',
            'merge sort': 'Thuật toán Merge Sort' if language == 'vi' else 'Merge Sort Algorithm',
            'bubble sort': 'Thuật toán Bubble Sort' if language == 'vi' else 'Bubble Sort Algorithm',
            'insertion sort': 'Thuật toán Insertion Sort' if language == 'vi' else 'Insertion Sort Algorithm',
            'selection sort': 'Thuật toán Selection Sort' if language == 'vi' else 'Selection Sort Algorithm',
            'heap sort': 'Thuật toán Heap Sort' if language == 'vi' else 'Heap Sort Algorithm',
            'radix sort': 'Thuật toán Radix Sort' if language == 'vi' else 'Radix Sort Algorithm',
            
            # Searching Algorithms
            'binary search': 'Thuật toán Tìm kiếm Nhị phân' if language == 'vi' else 'Binary Search Algorithm',
            'linear search': 'Thuật toán Tìm kiếm Tuyến tính' if language == 'vi' else 'Linear Search Algorithm',
            'depth first search': 'Tìm kiếm theo chiều sâu' if language == 'vi' else 'Depth-First Search',
            'breadth first search': 'Tìm kiếm theo chiều rộng' if language == 'vi' else 'Breadth-First Search',
            
            # Advanced Algorithms
            'dynamic programming': 'Quy hoạch động' if language == 'vi' else 'Dynamic Programming',
            'greedy algorithm': 'Thuật toán tham lam' if language == 'vi' else 'Greedy Algorithm',
            'backtracking': 'Thuật toán quay lui' if language == 'vi' else 'Backtracking Algorithm',
            'divide and conquer': 'Chia để trị' if language == 'vi' else 'Divide and Conquer',
            'dijkstra': 'Thuật toán Dijkstra' if language == 'vi' else 'Dijkstra\'s Algorithm',
            'a* algorithm': 'Thuật toán A*' if language == 'vi' else 'A* Search Algorithm',
        }
        
        for key, title in topic_mapping.items():
            if key in query.lower():
                conversation_name = title
                break
                
        # If no specific topic found, use a generic title
        if not conversation_name:
            conversation_name = "Cuộc hội thoại chưa có tiêu đề" if language == 'vi' else "Untitled Conversation"
    
    return conversation_name