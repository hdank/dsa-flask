from flask import jsonify, request
from app.core.utils import manage_conversation, get_conversation_history

def start_new_conversation():
    json_content = request.json
    user_id = json_content.get("user_id")
    conversation_id = manage_conversation(None, user_id)  # Force new conversation
    return jsonify({
        "status": "success",
        "conversation_id": conversation_id,
        "message": "New conversation started"
    })

def get_conversation_history_api(conversation_id):
    history = get_conversation_history(conversation_id)
    return jsonify({
        "conversation_id": conversation_id,
        "history": history
    })