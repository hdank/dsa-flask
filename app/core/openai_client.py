# app/core/openai_client.py
import requests
import logging
import json
from typing import List, Dict, Any, Iterator

logger = logging.getLogger(__name__)

class OpenAICompatibleClient:
    """Client for interacting with OpenAI-compatible API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url
        logger.info(f"OpenAI compatible client initialized with base URL: {base_url}")
        
    def chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        model: str, 
        temperature: float = 0.7, 
        stream: bool = True,
        functions: List[Dict[str, Any]] = None,
        function_call: Any = None
    ) -> Iterator[Dict[str, Any]]:
        """Send a chat completion request to the OpenAI-compatible API."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "max_tokens": 1024  # Set a reasonable limit
        }
        
        # Add function calling if provided
        if functions:
            payload["functions"] = functions
            if function_call:
                payload["function_call"] = function_call
        
        logger.info(f"Sending request to {url} with model {model}")
        
        try:
            # Use a timeout to prevent hanging requests
            with requests.post(url, json=payload, stream=True, timeout=30) as response:
                if response.status_code != 200:
                    error_msg = f"API request failed with status code {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    yield {
                        'message': {
                            'content': f"Error: {error_msg}",
                            'role': 'assistant'
                        }
                    }
                    return
                
                # Parse the streaming response
                if stream:
                    # Process streaming response
                    for line in response.iter_lines(chunk_size=8192):  # Increase chunk size for efficiency
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data:'):
                                data_str = line[5:].strip()
                                if data_str != "[DONE]":  # OpenAI sends "[DONE]" as the last message
                                    try:
                                        data = json.loads(data_str)
                                        if 'choices' in data and len(data['choices']) > 0:
                                            choice = data['choices'][0]
                                            if 'delta' in choice and 'content' in choice['delta'] and choice['delta']['content'] is not None:
                                                yield {
                                                    'message': {
                                                        'content': choice['delta']['content'],
                                                        'role': choice.get('delta', {}).get('role', 'assistant')
                                                    }
                                                }
                                            elif 'delta' in choice and 'function_call' in choice['delta']:
                                                # Handle function calling
                                                function_call_data = choice['delta']['function_call']
                                                yield {
                                                    'function_call': function_call_data
                                                }
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Error parsing JSON: {e}")
                                        logger.error(f"Problematic data: {data_str}")
                else:
                    # Handle non-streaming response
                    response_data = response.json()
                    yield {
                        'message': {
                            'content': response_data['choices'][0]['message']['content'],
                            'role': 'assistant'
                        }
                    }
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            yield {
                'message': {
                    'content': "Error: Request to the model timed out. Please try again.",
                    'role': 'assistant'
                }
            }
        except Exception as e:
            logger.error(f"Error making request: {e}")
            yield {
                'message': {
                    'content': f"Error: {str(e)}",
                    'role': 'assistant'
                }
            }