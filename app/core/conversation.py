# utils.py
import os
import json
import time
import uuid
from pathlib import Path
import logging
import asyncio
import threading
from app.core.config import CONVERSATION_TIMEOUT

# Setup base directory for conversations
CONVERSATIONS_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "conversations"))
CONVERSATIONS_DIR.mkdir(exist_ok=True)

def generate_conversation_id():
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())

def get_conversation_path(conversation_id):
    """Get the file path for a conversation JSON file."""
    return CONVERSATIONS_DIR / f"{conversation_id}.json"

def list_conversations():
    """Return a list of all conversation IDs"""
    conversations_path = CONVERSATIONS_DIR
    if not conversations_path.exists():
        return []
        
    conversation_files = conversations_path.glob("*.json")
    conversation_ids = [f.stem for f in conversation_files]
    return conversation_ids

def load_conversation(conversation_id):
    """Load conversation data from a JSON file."""
    path = get_conversation_path(conversation_id)
    logging.info(f"Attempting to load conversation from: {path}")
    
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logging.info(f"Successfully loaded conversation {conversation_id}")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading conversation {conversation_id}: {e}")
            return None
    else:
        logging.warning(f"Conversation file does not exist: {path}")
        return None

def save_conversation(conversation_id, data):
    """Save conversation data to JSON file."""
    if not conversation_id:
        conversation_id = generate_conversation_id()
    
    file_path = get_conversation_path(conversation_id)
    
    # Ensure last_activity is updated
    data["last_activity"] = time.time()
    
    # Create file with proper JSON structure
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return conversation_id

async def async_save_conversation(conversation_id, data):
    """Save conversation data asynchronously."""
    path = get_conversation_path(conversation_id)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logging.info(f"Error saving conversation {conversation_id}: {e}")

def delete_conversation(conversation_id):
    """Delete a conversation file."""
    if not conversation_id:
        return False
    
    file_path = get_conversation_path(conversation_id)
    if file_path.exists():
        os.remove(file_path)
        logging.info(f"Deleted this conversation_id: {conversation_id}")
        return True
    return False

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

def manage_conversation(conversation_id=None):
    """Create a new conversation or load an existing one."""
    if conversation_id is None:
        # Create new conversation
        conversation_id = generate_conversation_id()
        conversation_data = {
            "history": [],
            "created_at": time.time(),
            "last_activity": time.time(),
            "name": "New Conversation",
            "messages": []
        }
        save_conversation(conversation_id, conversation_data)
    else:
        # Verify the conversation exists and is not expired
        data = get_conversation_history(conversation_id)
        if data is None:
            # Create new conversation if the requested one doesn't exist
            conversation_id = generate_conversation_id()
            conversation_data = {
                "history": [],
                "created_at": time.time(),
                "last_activity": time.time(),
                "name": "New Conversation",
                "messages": []
            }
            save_conversation(conversation_id, conversation_data)
        elif (time.time() - data["last_activity"]) > CONVERSATION_TIMEOUT:
            # Create new conversation if the existing one is expired
            conversation_id = generate_conversation_id()
            conversation_data = {
                "history": [],
                "created_at": time.time(),
                "last_activity": time.time(),
                "name": "New Conversation",
                "messages": []
            }
            save_conversation(conversation_id, conversation_data)
            
    return conversation_id

def get_conversation_history(conversation_id):
    """Retrieve conversation history from file."""
    if not conversation_id:
        return None
    
    file_path = get_conversation_path(conversation_id)
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Update last_activity timestamp
        data["last_activity"] = time.time()
        save_conversation(conversation_id, data)
        
        return data
    except Exception as e:
        logging.error(f"Error loading conversation {conversation_id}: {e}")
        return None

def cleanup_old_conversations():
    """Remove expired conversations."""
    now = time.time()
    count = 0
    
    for file_path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if (now - data.get("last_activity", 0)) > CONVERSATION_TIMEOUT:
                os.remove(file_path)
                count += 1
        except Exception as e:
            logging.error(f"Error cleaning up conversation {file_path}: {e}")
    
    return count