# utils.py
import os
import json
import time
import uuid
import logging
import asyncio
import threading
from app.core.config import CONVERSATION_TIMEOUT

def get_conversation_path(conversation_id):
    """Get the file path for a conversation JSON file."""
    return os.path.join('conversations', f'{conversation_id}.json')

def load_conversation(conversation_id):
    """Load conversation data from a JSON file."""
    path = get_conversation_path(conversation_id)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.info(f"Error loading conversation {conversation_id}: {e}")
            return None
    return None

def save_conversation(conversation_id, data):
    """Save conversation data to a JSON file."""
    path = get_conversation_path(conversation_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logging.info(f"Error saving conversation {conversation_id}: {e}")

async def async_save_conversation(conversation_id, data):
    """Save conversation data asynchronously."""
    path = get_conversation_path(conversation_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logging.info(f"Error saving conversation {conversation_id}: {e}")

def delete_conversation(conversation_id):
    path = get_conversation_path(conversation_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        os.remove(path)
        logging.info(f"Deleted this conversation_id: {conversation_id}")
    except OSError as e:
        logging.info(f"Error deleting {conversation_id}: {e}")

def run_async(func, *args, **kwargs):
    """Run a function asynchronously in a separate thread."""
    def wrapper():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(func(*args, **kwargs))
        loop.close()
    
    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    return thread

def manage_conversation(conversation_id):
    """Manage conversation lifecycle using JSON files."""
    current_time = time.time()
    
    if conversation_id:
        data = load_conversation(conversation_id)
        if data:
            data['last_activity'] = current_time
            save_conversation(conversation_id, data)
            return conversation_id
        else:
            logging.info(f"Conversation {conversation_id} not found, creating new.")
    
    new_conv_id = str(uuid.uuid4())
    data = {
        'history': [],
        'created_at': current_time,
        'last_activity': current_time
    }
    save_conversation(new_conv_id, data)
    return new_conv_id

def get_conversation_history(conversation_id):
    """Retrieve conversation history from JSON file."""
    data = load_conversation(conversation_id)
    return data

def cleanup_old_conversations():
    """Delete conversations inactive beyond the timeout period."""
    current_time = time.time()
    timeout = CONVERSATION_TIMEOUT
    
    os.makedirs('conversations', exist_ok=True)
    
    for filename in os.listdir('conversations'):
        if filename.endswith('.json'):
            conv_id = filename[:-5]
            data = load_conversation(conv_id)
            if not data:
                continue
            if current_time - data['last_activity'] > timeout:
                try:
                    os.remove(get_conversation_path(conv_id))
                    logging.info(f"Cleaned up inactive conversation: {conv_id}")
                except OSError as e:
                    logging.info(f"Error deleting {conv_id}: {e}")