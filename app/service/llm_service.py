import logging, time, json, traceback, threading
from flask import Response
from app.core.utils import async_save_conversation, manage_conversation, cleanup_old_conversations, get_conversation_history, run_async, save_conversation
from app.core.naming import generate_conversation_name
from app.api.chat import rag_streaming_response
from app.core.vector_store import get_vector_store, retrieve_relevant_documents
from app.core.config import (
    OPENAI_API_BASE_URL, 
    OPENAI_MODEL,
    OPENAI_VISION_MODEL,
    SYSTEM_PROMPT
)
from app.core.openai_client import OpenAICompatibleClient
from app.core.evaluation import evaluate_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the OpenAI compatible client
openai_client = OpenAICompatibleClient(OPENAI_API_BASE_URL)

def ensure_structured_response(response, query):
    """Ensures the response follows the structured format with HTML tags"""
    # Check if the response contains any of the expected tags
    expected_tags = ["<CONCEPT>", "<EXAMPLE>", "<VISUALIZATION>", 
                     "<IMPLEMENTATION>", "<EXPLAINATION>", "<COMPLEXITY>"]
    
    has_tags = any(tag in response for tag in expected_tags)
    
    # Remove duplicate video sections before processing
    if "### Educational Videos" in response:
        # Get all parts of the text
        parts = response.split("### Educational Videos")
        
        # If there are multiple video sections, keep only the first one
        if len(parts) > 2:  # More than one "### Educational Videos" found
            base_text = parts[0]
            video_section = "### Educational Videos" + parts[1]
            response = base_text + video_section
    
    if not has_tags:
        # If it's likely about a concept
        if any(keyword in query.lower() for keyword in ["là gì", "khái niệm", "định nghĩa"]):
            # Create parts separately to avoid backslash issues in f-strings
            # Split by video section first if it exists
            main_content = response.split("### Educational Videos")[0] if "### Educational Videos" in response else response
            
            # Now split the main content for concept and example
            content_parts = main_content.split('\n\n')
            concept_part = content_parts[0] if content_parts else main_content
            example_part = content_parts[1] if len(content_parts) > 1 else "Xem phần khái niệm."
            
            structured = "<CONCEPT>\n"
            structured += concept_part
            structured += "\n</CONCEPT>\n\n"
            
            # Get video section if it exists
            video_section = ""
            if "### Educational Videos" in response:
                video_content = response.split("### Educational Videos")[1]
                video_content = video_content.replace("</VIDEOS>", "")
                video_section = "<VIDEOS>\n### Educational Videos" + video_content + "</VIDEOS>"
            
            # If we have videos, use them as the example
            if video_section:
                structured += "<EXAMPLE>\n"
                structured += video_section
                structured += "\n</EXAMPLE>\n\n"
            else:
                structured += "<EXAMPLE>\n"
                structured += example_part
                structured += "\n</EXAMPLE>\n\n"
            
            structured += "<VISUALIZATION>\n"
            structured += "Thuật toán này có thể được trực quan hóa thông qua các bước sắp xếp trên một mảng dữ liệu."
            structured += "\n</VISUALIZATION>"
            
            return structured
    
    # Fix potential issues with existing tags
    if "</VIDEOS></VIDEOS>" in response:
        response = response.replace("</VIDEOS></VIDEOS>", "</VIDEOS>")
    
    # Check for duplicate video sections in responses with tags
    if response.count("<VIDEOS>") > 1:
        # Keep only the first video section
        first_video_end = response.find("</VIDEOS>") + 9  # Length of "</VIDEOS>"
        second_video_start = response.find("<VIDEOS>", first_video_end)
        
        if second_video_start > 0:
            response = response[:second_video_start]
    
    return response

def _process_llama_request(form_data, image_base64=None):
    """
    Process request with or without image using OpenAI compatible API.
    
    Args:
        form_data: The form data from the request
        image_base64: Optional base64-encoded image
        
    Returns:
        Response: Server-sent events stream
    """
    query = form_data['query']
    conversation_id = form_data.get('conversation_id')
    
    # Manage conversation lifecycle
    conversation_id = manage_conversation(conversation_id)
    logging.info(f"Using conversation ID: {conversation_id}")
    
    # Periodically cleanup old conversations
    cleanup_old_conversations()
    
    return Response(
        _generate_llama_response(query, conversation_id, image_base64),
        content_type='text/event-stream; charset=utf-8'
    )

def _generate_llama_response(query, conversation_id, image_base64=None):
    """Generate streaming response using OpenAI compatible API with RAG integration."""
    from app.core.function_call import get_educational_video
    
    # Get existing conversation history
    history = get_conversation_history(conversation_id)
    
    # Choose the model based on whether an image is provided
    model_name = OPENAI_VISION_MODEL if image_base64 else OPENAI_MODEL
    
    logging.info(f"Processing with model: {model_name}")
    
    try:
        # Generate conversation name if it doesn't exist
        if not history or not history.get('name') or history.get('name') == "New Conversation":
            try:
                current_history = history.get('messages', []) if history else []
                history = history or {}
                history['name'] = generate_conversation_name(current_history, query)
            except Exception as e:
                logging.error(f"Error generating conversation name: {str(e)}")
            run_async(async_save_conversation, conversation_id, history)
        
        # Add user message to history first
        if 'messages' not in history:
            history['messages'] = []
            
        # Store the user's query
        user_message = {
            'role': 'user',
            'content': query,
            'created_at': time.time()  
        }
        
        # If it's an image, store that in the message
        if image_base64:
            # Store a reference that this message had an image
            # (don't store the full base64 in history to save space)
            user_message['has_image'] = True
        
        # Add to history
        history['messages'].append(user_message)
        
        # Get retrieved documents for RAG - now for both text and image queries
        retrieved_docs = []
        vector_store = get_vector_store()
        retrieved_docs = retrieve_relevant_documents(query, vector_store)
        logging.info(f"Retrieved {len(retrieved_docs)} documents for RAG")
        
        # Proactively check for algorithm-related topics and add videos
        lower_query = query.lower()
        force_video_for_topics = [
            "sort", "merge sort", "quick sort", "bubble sort", "insertion sort",
            "selection sort", "heap sort", "shell sort", "radix sort", "binary search",
            "depth first search", "breadth first search", "dfs", "bfs", "queue",
            "stack", "linked list", "tree", "graph", "hash"
        ]
        
        # If it's a query about algorithms/data structures, proactively add videos
        should_add_videos = any(topic in lower_query for topic in force_video_for_topics)
        video_topic = None
        
        if should_add_videos:
            # Determine the most specific topic match
            for topic in sorted(force_video_for_topics, key=len, reverse=True):  # Sort by length to match most specific first
                if topic in lower_query:
                    video_topic = topic
                    break
            
            if not video_topic:
                video_topic = next((topic for topic in force_video_for_topics if topic in lower_query), None)
        
        # Format messages for the model
        if image_base64:
            # Vision model format - now with RAG
            # Create a prompt that includes context from RAG
            system_message = "You are an assistant that can see images and analyze them. "
            
            # If we have retrieved documents, include them as context
            if retrieved_docs:
                system_message += "Here is some additional context that might be helpful:\n\n"
                for i, doc in enumerate(retrieved_docs):
                    system_message += f"Document {i+1}:\n{doc.get('content', '')}\n\n"
            
            messages = [
                {
                    'role': 'system',
                    'content': system_message
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': query
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
        else:
            # Text model format with RAG and improved system prompt
            # Format the conversation history
            formatted_history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in history.get('messages', [])
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg
                and msg['role'] != 'assistant'  # Skip past assistant responses to avoid repetition
            ])
            
            # Format retrieved context for the prompt
            context_str = "\n".join([
                f"Tài liệu: {doc.get('content', '')}\nĐộ liên quan: {doc.get('score', 0)}" 
                for doc in retrieved_docs
            ]) if retrieved_docs else "Không tìm thấy ngữ cảnh liên quan."
            
            # Format the prompt with the retrieved context, history, and query
            messages = [
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT.format(
                        context=context_str,
                        history=formatted_history,
                        input=query
                    )
                }
            ]
        
        # Set timeout for the API request to prevent hanging
        response_data = {"completed": False, "error": None, "chunks": []}
        
        # This function will be executed in a separate thread
        def process_stream():
            try:
                logging.info(f"Sending request to API: {OPENAI_API_BASE_URL}")
                
                # Use the direct API request approach to avoid potential client issues
                import requests
                
                # Construct URL for the API endpoint
                url = f"{OPENAI_API_BASE_URL}/chat/completions"
                
                # Create API payload
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7
                }
                
                # Make streaming API request
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=True
                )
                
                if not response.ok:
                    logging.error(f"API request failed with status code {response.status_code}: {response.text}")
                    response_data["error"] = f"API request failed: {response.text}"
                    return
                
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data.strip() == '[DONE]':
                                break
                                
                            try:
                                chunk_data = json.loads(data)
                                if chunk_data.get('choices') and len(chunk_data['choices']) > 0:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    
                                    if content:
                                        response_data["chunks"].append(content)
                            except json.JSONDecodeError as e:
                                logging.error(f"Failed to parse JSON chunk: {data}, error: {e}")
                
                response_data["completed"] = True
                
            except Exception as e:
                logging.error(f"Error in process_stream: {str(e)}")
                logging.error(traceback.format_exc())
                response_data["error"] = str(e)
        
        # Start processing in a separate thread with timeout
        process_thread = threading.Thread(target=process_stream)
        process_thread.daemon = True
        process_thread.start()
        
        # Define a reasonable timeout (e.g., 60 seconds)
        timeout_seconds = 60
        start_time = time.time()
        
        def generate():
            full_response = ""
            is_new_conversation = len(history.get('messages', [])) <= 1  # Only user message exists
            
            # Send retrieved documents as the initial message - for both text and vision queries
            if retrieved_docs:
                # Format and send document information
                doc_info = {
                    'type': 'context',
                    'conversation_id': conversation_id,
                    'documents': [
                        {
                            'content': doc.get('content', '')[:200] + '...',  # First 200 chars
                            'metadata': doc.get('metadata', {}),
                            'score': float(doc.get('score', 0))  # Ensure score is JSON serializable
                        }
                        for doc in retrieved_docs
                    ]
                }
                yield f"data: {json.dumps(doc_info)}\n\n"
            
            # Check for thread completion or timeout
            while not response_data["completed"] and not response_data["error"] and (time.time() - start_time < timeout_seconds):
                # Yield chunks as they come in
                while response_data["chunks"]:
                    chunk = response_data["chunks"].pop(0)
                    full_response += chunk
                    
                    data = {
                        'type': 'content',
                        'chunk': chunk,
                        'full': full_response,
                        'conversation_id': conversation_id,
                        'conversation_name': history.get('name', ''),
                        'is_new_conversation': is_new_conversation,
                        # Include docs if they exist
                        'docs': [
                            {
                                'content': doc.get('content', '')[:100],  # Preview
                                'metadata': doc.get('metadata', {}),
                                'score': float(doc.get('score', 0))
                            }
                            for doc in retrieved_docs
                        ] if retrieved_docs else []
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                
                # Small sleep to prevent tight loop
                time.sleep(0.1)
            
            # Check if we timed out or had an error
            if time.time() - start_time >= timeout_seconds and not response_data["completed"]:
                error_data = {
                    'error': "Request timed out",
                    'conversation_id': conversation_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            elif response_data["error"]:
                error_data = {
                    'error': response_data["error"],
                    'conversation_id': conversation_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            
            # After the model has finished generating, check if we should add videos if none were added
            if full_response and should_add_videos and video_topic:
                try:
                    # Use the original query to find the best video match
                    from app.core.function_call import get_educational_video
                    
                    # Log the exact topic we're searching for
                    logging.info(f"Searching videos for topic: {query}")
                    
                    # Try to match with the full query first
                    videos = get_educational_video(query, 3)
                    
                    # If no videos found with full query, try with the identified topic
                    if not videos and video_topic:
                        videos = get_educational_video(video_topic, 3)
                    
                    if videos:
                        # Format video section - use the actual topic from the query for better heading
                        search_topic = query if len(query) < 30 else video_topic
                        video_section = f"\n\n<VIDEOS>\n### Educational Videos for {search_topic.title()}:\n"
                        for idx, video in enumerate(videos, 1):
                            video_section += f"{idx}. [{video['title']}]({video['url']})\n"
                        video_section += "</VIDEOS>"
                        
                        # Add to full response
                        full_response += video_section
                        
                        # Send final chunk with videos
                        data = {
                            'type': 'content',
                            'chunk': video_section,
                            'full': full_response,
                            'conversation_id': conversation_id,
                            'conversation_name': history.get('name', ''),
                            'is_new_conversation': is_new_conversation
                        }
                        
                        yield f"data: {json.dumps(data)}\n\n"
                except Exception as e:
                    logging.error(f"Error adding educational videos: {str(e)}")
                    logging.error(traceback.format_exc())
            
            # Save the complete conversation
            if full_response:
                # Apply structure formatting
                full_response = ensure_structured_response(full_response, query)
            # Evaluate the response
            try:
                evaluation, eval_path = evaluate_response(query, full_response, video_topic, conversation_id)
                logger.info(f"Response evaluation score: {evaluation['combined_score']}")
                
                # Create assistant message with embedded evaluation
                assistant_message = {
                    'role': 'assistant',
                    'model': model_name,
                    'content': full_response,
                    'created_at': time.time(),
                    'evaluation': {
                        'scores': {
                            'structure': evaluation['structure_score'],
                            'content': evaluation['content_score'],
                            'relevance': evaluation['relevance_score'],
                            'combined': evaluation['combined_score']
                        },
                        'summary': {
                            'structure': evaluation['details']['structure'].get('summary', ''),
                            'content': evaluation['details']['content'].get('summary', ''),
                            'relevance': evaluation['details']['relevance'].get('summary', '')
                        },
                        'summary_vi': {
                            'structure': evaluation['details']['structure'].get('summary_vi', ''),
                            'content': evaluation['details']['content'].get('summary_vi', ''),
                            'relevance': evaluation['details']['relevance'].get('summary_vi', '')
                        },
                        'findings': {
                            'structure': evaluation['details']['structure'].get('findings', []),
                            'content': evaluation['details']['content'].get('findings', []),
                            'relevance': evaluation['details']['relevance'].get('findings', [])
                        },
                        'findings_vi': {
                            'structure': evaluation['details']['structure'].get('findings_vi', []),
                            'content': evaluation['details']['content'].get('findings_vi', []),
                            'relevance': evaluation['details']['relevance'].get('findings_vi', [])
                        },
                        'eval_file_path': eval_path
                    }
                }
                
                # Always store retrieved documents in the message for future reference
                if retrieved_docs:
                    # Store full document info in the conversation history
                    assistant_message['docs'] = [
                        {
                            'content': doc.get('content', ''),  # Store full content in history
                            'metadata': doc.get('metadata', {}),
                            'score': float(doc.get('score', 0))
                        }
                        for doc in retrieved_docs
                    ]
                
                # Add to history
                history['messages'].append(assistant_message)
                
                # Update last activity timestamp
                history['last_activity'] = time.time()
                
                # Save the complete conversation
                run_async(async_save_conversation, conversation_id, history)
                logging.info(f"Saved conversation {conversation_id} with {len(history['messages'])} messages")
                
                # Send full evaluation data to the client
                eval_data = {
                    'type': 'evaluation',
                    'conversation_id': conversation_id,
                    'scores': assistant_message['evaluation']['scores'],
                    'summary': assistant_message['evaluation']['summary'],
                    'summary_vi': assistant_message['evaluation']['summary_vi'],
                    'findings': assistant_message['evaluation']['findings'],
                    'findings_vi': assistant_message['evaluation']['findings_vi'],
                    'eval_path': eval_path
                }
                yield f"data: {json.dumps(eval_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Error evaluating response: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Still create assistant message but without evaluation
                assistant_message = {
                    'role': 'assistant',
                    'model': model_name,
                    'content': full_response,
                    'created_at': time.time()
                }
                
                # Add docs if available
                if retrieved_docs:
                    assistant_message['docs'] = [
                        {
                            'content': doc.get('content', ''),
                            'metadata': doc.get('metadata', {}),
                            'score': float(doc.get('score', 0))
                        }
                        for doc in retrieved_docs
                    ]
                
                # Add to history
                history['messages'].append(assistant_message)
                history['last_activity'] = time.time()
                run_async(async_save_conversation, conversation_id, history)
                
            yield "data: [DONE]\n\n"
        
        return generate()
    except Exception as e:
        logging.error(f"Error in _generate_llama_response: {str(e)}")
        logging.error(traceback.format_exc())
        
        def generate_error():
            error_data = {
                'error': str(e),
                'conversation_id': conversation_id
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        
        return generate_error()

def _build_prompt(system_prompt, history, query):
    """
    Build full prompt with system instructions, history, and query.
    
    Args:
        system_prompt: System instructions
        history: Conversation history
        query: Current user query
        
    Returns:
        str: Formatted prompt
    """
    # The system_prompt already contains all the necessary instructions, context, and the query
    logging.info("Building prompt with system instructions and query")
    return system_prompt

def _encode_sse_message(data, conversation_id, is_new_conversation):
    """Encode a message for Server-Sent Events."""
    data.update({
        'conversation_id': conversation_id,
        'is_new_conversation': is_new_conversation
    })
    return f"data: {json.dumps(data)}\n\n"