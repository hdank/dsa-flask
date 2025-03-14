import logging
from flask import request, jsonify, send_from_directory
from langchain_community.document_loaders import PDFPlumberLoader
from app.core.vector_store import store_documents, delete_document_by_id
from app.core.config import PDF_FOLDER
import os, base64
from app.service.llm_service import _process_llama_request

def open_pdf_to_web_browser():
    json_content = request.json
    file_name = json_content.get("file_name") + ".pdf"

    if os.path.exists(os.path.join(PDF_FOLDER, file_name)):
        pdf_url = f"{request.host_url}pdfs/{file_name}"  # Construct the public URL
        return jsonify({"pdf_url": pdf_url}), 200
    else:
        return jsonify({"error": "File not found"}), 404
    
def serve_pdf(filename):
    return send_from_directory(PDF_FOLDER, filename)

def ask_text():
    """Handle requests to the /ask_llama endpoint."""
    logging.info("POST /ask_llama called")
    return _process_llama_request(request.form, None)

def ask_vision():
    """Handle requests to the /ask_llama_vision endpoint."""
    logging.info("POST /ask_llama_vision called")
    image_file = request.files.get('image')
    
    # Check if query is provided
    if 'query' not in request.form:
        return jsonify({"error": "No query provided"}), 400
    
    # Process image if provided
    image_base64 = None
    if image_file:
        try:
            # Resize the image before encoding to reduce size
            from PIL import Image
            import io
            
            image_data = image_file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Resize to a reasonable size (maintain aspect ratio)
            max_size = 800
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
            
            # Convert back to bytes - USE THIS RESIZED IMAGE DATA
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logging.info(f"Image processed successfully, size: {len(image_base64)}")
            return _process_llama_request(request.form, image_base64)
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    else:
        return jsonify({"error": "No image file provided"}), 400

def pdf_post():
    file = request.files["file"]
    file_name = file.filename
    save_file = os.path.join(PDF_FOLDER, file_name)
    file.save(save_file)
    logging.info(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    logging.info(f"docs len={len(docs)}")

    document_id = store_documents(docs)

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "document_id": document_id  # Return the auto-generated IDs
    }
    return response

def delete_pdf():
    try:
        json_content = request.json
        file_name = json_content.get("file_name")+".pdf"
        document_id = json_content.get("document_id")

        # Delete the document and its chunks from vector store
        delete_document_by_id(document_id)

        # Delete the PDF file if it exists
        save_file = os.path.join(PDF_FOLDER, file_name)
        if os.path.exists(save_file):
            os.remove(save_file)
            logging.info(f"Deleted PDF file: {save_file}")

        return {
            "status": 200,
            "message": f"Successfully deleted document {document_id} and file {file_name}"
        }
        
    except Exception as e:
        return {
            "status": 500,
            "error": f"Error deleting document: {str(e)}"
        }, 500
