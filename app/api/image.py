from flask import request, jsonify
from app.core.llm import stream_chat_response
import base64

def ask_image_post():
    print("POST /ask_image called")
    
    # Check if the image file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    
    # Check if the query is in the request
    if 'query' not in request.form:
        return jsonify({"error": "No query provided"}), 400
    
    query = request.form['query']

    # Read the image file and convert it to base64
    image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    messages = [
        {'role': 'user',
         'content': query,
         'images': [image_base64] 
        }
    ]
    
    res = stream_chat_response(messages)
    
    full_response = ""
    for chunk in res:
        full_response += chunk['message']['content']
    print(full_response)
    return jsonify({"response": full_response})