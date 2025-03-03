import logging, time, json
from flask import Response
from app.core.utils import async_save_conversation, manage_conversation, cleanup_old_conversations, get_conversation_history, run_async, save_conversation
from app.core.naming import generate_conversation_name
from app.api.chat import rag_streaming_response
from app.core.vector_store import get_vector_store, retrieve_relevant_documents
from app.core.config import (
    OPENAI_API_BASE_URL, 
    OPENAI_MODEL,  # "llama-3.2-1b-instruct:2"
    OPENAI_VISION_MODEL  # "qwen2-vl-7b-instruct"
)
from app.core.openai_client import OpenAICompatibleClient

# Initialize the OpenAI compatible client
openai_client = OpenAICompatibleClient(OPENAI_API_BASE_URL)

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
    import traceback
    import time
    import json
    import threading
    from app.core.openai_client import OpenAICompatibleClient
    from app.core.config import OPENAI_API_BASE_URL, OPENAI_MODEL, OPENAI_VISION_MODEL
    
    # Initialize the OpenAI client
    openai_client = OpenAICompatibleClient(OPENAI_API_BASE_URL)
    
    # Get existing conversation history
    history = get_conversation_history(conversation_id)
    
    # Choose the model based on whether an image is provided
    model_name = OPENAI_VISION_MODEL if image_base64 else OPENAI_MODEL
    
    logging.info(f"Processing with model: {model_name}")
    
    try:
        # Generate conversation name if it doesn't exist
        if not history.get('name'):
            name_suffix = " (image analysis)" if image_base64 else ""
            try:
                current_history = history.get('messages', [])
                history['name'] = generate_conversation_name(current_history, query)
            except Exception as e:
                logging.error(f"Error generating conversation name: {str(e)}")
                history['name'] = f"Untitled"
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
            
            # No function calling for vision model
            function_call = None
            functions = None
        else:
            # Text model format with function calling and RAG
            messages = [
                {
                    'role': 'system', 
                    'content': "You are a technical assistant specialized in data structures and algorithms."
                },
                {
                    'role': 'user', 
                    'content': rag_streaming_response(query, history.get('messages', []))
                }
            ]
            
            # Define function schema for text model
            functions = [
                {
                    "name": "get_educational_video",
                    "description": "Get specific, pre-selected educational videos about data structures and algorithms topics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string", 
                                "description": "The data structure or algorithm topic to find videos about (e.g., 'merge sort', 'binary tree')"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of video results to return",
                                "default": 3
                            }
                        },
                        "required": ["topic"]
                    }
                }
            ]
            
            # Force function calling for certain topics
            function_call = "auto"
            visualization_topics = [
                "sort", "search", "tree", "graph", "list", "stack", "queue", "heap",
                "hash", "binary", "merge", "quick", "insertion", "bubble", "linked list",
                "depth first", "breadth first", "dfs", "bfs", "algorithm", "data structure"
            ]
            
            if any(topic in lower_query for topic in visualization_topics):
                function_call = {"name": "get_educational_video"}
                logging.info(f"Forcing function call to get_educational_video for query: {query}")
        
        # Set timeout for the API request to prevent hanging
        response_data = {"completed": False, "error": None, "chunks": []}
        
        def process_stream():
            try:
                # Use the OpenAI client to get a response stream
                stream = openai_client.chat_completion(
                    messages=messages,
                    model=model_name,
                    functions=functions,
                    function_call=function_call
                )
                
                # Process the response stream
                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        response_data["chunks"].append(chunk['message']['content'])
                    elif 'function_call' in chunk:
                        # Handle function calling
                        response_data["chunks"].append(json.dumps(chunk['function_call']))
                
                response_data["completed"] = True
            except Exception as e:
                logging.error(f"Error in generate_response: {str(e)}")
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
                    
                    # Process function calls if present
                    function_call_data = None
                    try:
                        # Check if the chunk is a JSON string containing function call data
                        parsed = json.loads(chunk)
                        if isinstance(parsed, dict) and 'name' in parsed and 'arguments' in parsed:
                            function_call_data = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                        
                    if function_call_data:
                        try:
                            # Handle function call and generate response
                            function_name = function_call_data.get('name')
                            arguments = json.loads(function_call_data.get('arguments', '{}'))
                            
                            if function_name == 'get_educational_video':
                                topic = arguments.get('topic', '')
                                max_results = int(arguments.get('max_results', 3))
                                video_results = get_educational_video(topic, max_results)
                                
                                # Append video results to full response
                                video_response = f"\n\n### Educational Videos about {topic}:\n"
                                for idx, video in enumerate(video_results, 1):
                                    video_response += f"{idx}. [{video['title']}]({video['url']})\n"
                                    video_response += f"   Channel: {video['channel']}\n\n"
                                
                                full_response += video_response
                                chunk = video_response
                        except Exception as func_err:
                            logging.error(f"Error in function call: {str(func_err)}")
                    
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
                        video_section = f"\n\n### Educational Videos for {search_topic.title()}:\n"
                        for idx, video in enumerate(videos, 1):
                            video_section += f"{idx}. [{video['title']}]({video['url']})\n"
                            video_section += f"   Channel: {video['channel']}\n\n"
                        
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
                # Create assistant message
                assistant_message = {
                    'role': 'assistant',
                    'model': model_name,
                    'content': full_response,
                    'created_at': time.time()
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
                save_conversation(conversation_id, history)
                logging.info(f"Saved conversation {conversation_id} with {len(history['messages'])} messages")
        
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