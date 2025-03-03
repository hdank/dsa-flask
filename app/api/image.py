from flask import request, jsonify
import base64
from app.core.openai_client import OpenAICompatibleClient
from app.core.config import OPENAI_API_BASE_URL

def ask_image_post():
    print("POST /ask_image called")
    
    # Check if the image file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    
    # Check if the query is in the request
    if 'query' not in request.form:
        return jsonify({"error": "No query provided"}), 400
    
    # Pass to the vision API which now handles conversation storage
    from app.service.llm_service import _process_llama_request_with_qwen
    
    # Read the image file and convert it to base64
    image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Use the service function to properly handle conversation
    return _process_llama_request_with_qwen(request.form, image_base64)