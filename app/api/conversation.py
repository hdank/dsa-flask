from flask import jsonify, request
from app.core.conversation import manage_conversation, get_conversation_history, delete_conversation, save_conversation

def start_new_conversation():
    conversation_id = manage_conversation(None)  # Force new conversation
    return jsonify({
        "status": "success",
        "conversation_id": conversation_id,
        "message": "New conversation started"
    })

def delete_a_conversation(conversation_id):
    delete_conversation(conversation_id)
    return jsonify({
        "status": "success",
        "conversation_id": conversation_id,
        "message": "Successfully deleted this conversation"
    })

def get_conversation_history_api(conversation_id):
    """API endpoint to retrieve a conversation's history."""
    data = get_conversation_history(conversation_id)
    
    if data is None:
        return jsonify({
            "status": "error",
            "conversation_id": conversation_id,
            "message": "Conversation not found"
        }), 404
        
    return jsonify({
        "status": "success",
        "conversation_id": conversation_id,
        "data": data
    })

def update_conversation_api(conversation_id):
    """API endpoint to update a conversation's data"""
    json_content = request.json
    if not json_content:
        return jsonify({
            "error": "No data provided",
            "status": 400
        }), 400
    
    # Get existing data first
    existing_data = get_conversation_history(conversation_id)
    if not existing_data:
        return jsonify({
            "status": "error",
            "conversation_id": conversation_id,
            "message": "Conversation not found"
        }), 404
    
    # Update the conversation with new data
    updated_data = {**existing_data, **json_content}
    save_conversation(conversation_id, updated_data)
    
    return jsonify({
        "status": "success",
        "conversation_id": conversation_id,
        "message": "Conversation updated successfully"
    })

def get_conversations_history_api():
    """API endpoint to retrieve multiple conversation histories."""
    json_content = request.json
    conversation_ids = json_content.get("conversation_ids", [])
    
    if not conversation_ids:
        return jsonify({
            "error": "No conversation IDs provided",
            "status": 400
        }), 400
    
    # Get histories for all conversations
    conversations_data = []
    for conversation_id in conversation_ids:
        data = get_conversation_history(conversation_id)
        if data:
            conversations_data.append({
                "conversation_id": conversation_id,
                "data": data,
                "status": "success"
            })
        else:
            conversations_data.append({
                "conversation_id": conversation_id,
                "data": None,
                "status": "error",
                "message": f"Could not load conversation {conversation_id}"
            })
    
    return jsonify({
        "conversations": conversations_data,
        "total_conversations": len(conversations_data),
        "successful_loads": sum(1 for conv in conversations_data if conv["status"] == "success"),
        "failed_loads": sum(1 for conv in conversations_data if conv["status"] == "error")
    })