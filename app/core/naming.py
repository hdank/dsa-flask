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
            prompt = f"Tạo tiêu đề ngắn gọn cho cuộc trò chuyện về: \"{query}\""
        else:
            prompt = f"Create a short title for a conversation about: \"{query}\""
                
        # Call LLM only once
        conversation_name = generate_text_from_llm(prompt)
    except Exception as e:
        logging.error(f"Error using LLM for conversation name: {str(e)}")
    
    # Create topic-based titles if LLM fails
    if not conversation_name:
        # Check for common data structure/algorithm terms
        topic_mapping = {
            'stack': 'Cấu trúc dữ liệu Stack' if language == 'vi' else 'Stack Data Structure',
            'queue': 'Cấu trúc dữ liệu Queue' if language == 'vi' else 'Queue Data Structure',
            'linked list': 'Cấu trúc dữ liệu Linked List' if language == 'vi' else 'Linked List Data Structure',
            'binary tree': 'Cấu trúc dữ liệu Binary Tree' if language == 'vi' else 'Binary Tree Data Structure',
            'quick sort': 'Thuật toán Quick Sort' if language == 'vi' else 'Quick Sort Algorithm',
            'merge sort': 'Thuật toán Merge Sort' if language == 'vi' else 'Merge Sort Algorithm',
            'bubble sort': 'Thuật toán Bubble Sort' if language == 'vi' else 'Bubble Sort Algorithm',
        }
        
        for key, title in topic_mapping.items():
            if key in query.lower():
                conversation_name = title
                break
                
        # If no specific topic found, use a generic title
        if not conversation_name:
            conversation_name = "Cuộc hội thoại chưa có tiêu đề" if language == 'vi' else "Untitled Conversation"
    
    return conversation_name
