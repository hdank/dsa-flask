# app/__init__.py
from flask import Flask
from app.core.config import BASE_DIR
import os

def create_app():
    app = Flask(__name__)
    
    # Ensure the 'pdf' and 'db' directories exist
    os.makedirs(os.path.join(BASE_DIR, "pdf"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)
    
    # Register blueprints for routes
    from app.api.chat import ai_post
    from app.api.pdf import pdf_post, ask_text, ask_vision, delete_pdf, open_pdf_to_web_browser, serve_pdf
    from app.api.image import ask_image_post
    from app.api.conversation import start_new_conversation, get_conversation_history_api, get_conversations_history_api, delete_a_conversation
    
    # Add routes to the app
    app.add_url_rule("/ask_text", methods=["POST"], view_func=ask_text)
    app.add_url_rule("/ask_vision", methods=["POST"], view_func=ask_vision)
    app.add_url_rule("/delete-pdf", methods=["DELETE"], view_func=delete_pdf)
    app.add_url_rule("/open_pdf", methods=["POST"], view_func=open_pdf_to_web_browser)
    app.add_url_rule("/pdfs/<path:filename>", methods=["GET"], view_func=serve_pdf)
    app.add_url_rule("/ask_image", methods=["POST"], view_func=ask_image_post)
    app.add_url_rule("/new_conversation", methods=["POST"], view_func=start_new_conversation)
    app.add_url_rule("/conversation_history/<conversation_id>", methods=["GET"], view_func=get_conversation_history_api)
    app.add_url_rule("/conversations_history", methods=["POST"], view_func=get_conversations_history_api)
    app.add_url_rule("/delete_conversation/<conversation_id>", methods=["POST"], view_func=delete_a_conversation)

    return app