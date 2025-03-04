import requests
import json
import logging
from typing import Dict, Any, List, Optional, Union, Generator, Iterator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatCompletionChunk:
    """Class to represent a chunk of a streaming chat completion response"""
    def __init__(self, chunk_data):
        self.id = chunk_data.get("id", "")
        self.model = chunk_data.get("model", "")
        self.choices = [ChatCompletionChunkChoice(choice) for choice in chunk_data.get("choices", [])]

class ChatCompletionChunkChoice:
    """Class to represent a choice in a chat completion chunk"""
    def __init__(self, choice_data):
        self.index = choice_data.get("index", 0)
        self.delta = ChatCompletionChunkDelta(choice_data.get("delta", {}))
        self.finish_reason = choice_data.get("finish_reason", None)

class ChatCompletionChunkDelta:
    """Class to represent delta in a chat completion chunk choice"""
    def __init__(self, delta_data):
        self.role = delta_data.get("role", None)
        self.content = delta_data.get("content", None)

class ChatCompletions:
    """Class to handle chat completions API calls"""
    def __init__(self, client):
        self.client = client

    def create(self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[Dict[str, Any], Generator]:
        """Create a chat completion with the given parameters"""
        url = f"{self.client.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        try:
            if stream:
                return self._stream_response(url, payload, headers)
            else:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed with error: {str(e)}")
            raise

    def _stream_response(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Iterator[ChatCompletionChunk]:
        """Stream the response from the API"""
        try:
            response = requests.post(url, json=payload, headers=headers, stream=True)
            response.raise_for_status()
            
            # Process each line of the streaming response
            for line in response.iter_lines():
                if line:
                    # Skip empty or keep-alive lines
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            yield ChatCompletionChunk(chunk_data)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON chunk: {data}")
                            
        except requests.RequestException as e:
            logger.error(f"API request failed with error: {str(e)}")
            raise

class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs like Ollama"""
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.chat = ChatCompletions(self)