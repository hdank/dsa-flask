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
        print(f"Warning: Error generating text from LLM: {e}")
        return ""  # Return empty string as fallback

def generate_conversation_name(conversation_history: List[dict], query: str) -> str:
    """
    Generate a creative, professional conversation title based on the conversation context and query.
    The function auto-detects the language from the query and then builds a generative prompt for the LLM.
    """
    # Detect language from the query.
    language = detect_language(query)
    
    # Extract keywords based on language.
    if language == "vi":
        key_words = extract_keywords_vi(conversation_history, query)
    else:
        key_words = extract_keywords_en(conversation_history, query)
    
    # Prepare a generative prompt including the query and the extracted topics.
    if language == "vi":
        topics_str = ", ".join(key_words) if key_words else ""
        prompt = (
            f"Hãy tạo một tiêu đề cuộc trò chuyện sáng tạo và chuyên nghiệp cho nội dung: \"{query}\" "
            f"với các chủ đề chính: {topics_str}. Tiêu đề cần ngắn gọn, thu hút và phù hợp với văn phong tiếng Việt."
            f"Chỉ được đưa ra MỘT tiêu đề, không được nhiều hơn. Chỉ được phép trả lời với tên tiêu đề và không bình luận gì thêm. "
        )
    else:
        topics_str = ", ".join(key_words) if key_words else ""
        prompt = (
            f"Generate a creative and professional conversation title for a discussion with the content: \"{query}\" "
            f"and key topics: {topics_str}. The title should be short, engaging, and appropriate for a professional setting."
            f"Only generate ONE conversation title, do not commentary about anything else, you should only response with the conversation title"
        )
    
    # Call the LLM to generate the title.
    conversation_name = generate_text_from_llm(prompt)
    
    # Fallback if no title is generated.
    if not conversation_name:
        conversation_name = "Cuộc hội thoại chưa có tiêu đề" if language == "vi" else "Untitled Conversation"
    
    return conversation_name
