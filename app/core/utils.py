import time, uuid
from app.core.config import CONVERSATION_TIMEOUT

# Global dictionary to store conversation histories and their last activity timestamp
conversation_histories = {}
conversation_metadata = {}

def manage_conversation(conversation_id=None):
    """
    Manages conversation lifecycle. Returns appropriate conversation_id.
    Creates new conversation only when needed.
    """
    current_time = time.time()
    
    # If conversation_id is provided, check if it exists and is still valid
    if conversation_id:
        if conversation_id in conversation_metadata:
            # Update last activity timestamp
            conversation_metadata[conversation_id]['last_activity'] = current_time
            return conversation_id
        else:
            # If the ID doesn't exist, we'll create a new one
            print(f"Conversation {conversation_id} not found, creating new conversation")
    
    # Create new conversation
    new_conversation_id = str(uuid.uuid4())
    conversation_histories[new_conversation_id] = []
    conversation_metadata[new_conversation_id] = {
        'created_at': current_time,
        'last_activity': current_time
    }
    return new_conversation_id

def cleanup_old_conversations():
    """
    Removes conversations that have been inactive for more than 24 hours
    """
    current_time = time.time()
    timeout = CONVERSATION_TIMEOUT
    
    for conv_id in list(conversation_metadata.keys()):
        if current_time - conversation_metadata[conv_id]['last_activity'] > timeout:
            del conversation_histories[conv_id]
            del conversation_metadata[conv_id]
            print(f"Cleaned up inactive conversation: {conv_id}")

def get_conversation_history(conversation_id):
     return conversation_histories.get(conversation_id, [])