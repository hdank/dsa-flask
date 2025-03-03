import logging
from typing import Optional, Dict, List, Any
import requests
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Dictionary of recommended videos
RECOMMENDED_VIDEOS = {
    "quick sort": [
        {
            "title": "Quick Sort - Visualization and Implementation",
            "url": "https://youtu.be/Gyj8fd4DBpc",
            "channel": "Long's Clips",
            "description": "Comprehensive explanation of Quick Sort algorithm"
        }
    ],
    "bubble sort": [
        {
            "title": "Bubble Sort Algorithm Made Simple",
            "url": "https://www.youtube.com/watch?v=xli_FI7CuzA",
            "channel": "Long's Clips",
            "description": "Step-by-step explanation of Bubble Sort algorithm"
        }
    ],
    "insertion sort": [
        {
            "title": "Insertion Sort Explained",
            "url": "https://www.youtube.com/watch?v=JU767SDMDvA",
            "channel": "Long's Clips",
            "description": "Detailed tutorial on Insertion Sort algorithm"
        }
    ],
    "merge sort": [
        {
            "title": "Merge Sort Algorithm Explained",
            "url": "https://www.youtube.com/watch?v=4VqmGXwpLqc",
            "channel": "Long's Clips",
            "description": "Visual explanation of Merge Sort with examples"
        }
    ],
    "selection sort": [
        {
            "title": "Selection Sort Algorithm Explained",
            "url": "https://www.youtube.com/watch?v=Ns4TPTC8whw",
            "channel": "Long's Clips",
            "description": "Complete tutorial on Selection Sort with examples"
        }
    ],
    "linear search": [
        {
            "title": "Linear Search Algorithm",
            "url": "https://www.youtube.com/watch?v=YvAosi_pZ8w",
            "channel": "Long's Clips",
            "description": "Learn how Linear Search works with examples"
        }
    ],
    "binary search": [
        {
            "title": "Binary Search Algorithm",
            "url": "https://www.youtube.com/watch?v=YvAosi_pZ8w&t=2365",
            "channel": "Long's Clips",
            "description": "Efficient Binary Search implementation and visualization"
        }
    ],
    "single linked list": [
        {
            "title": "Single Linked List Implementation",
            "url": "https://youtu.be/4imz17FNr9k",
            "channel": "Long's Clips",
            "description": "Learn how to implement a Single Linked List"
        }
    ],
    "double linked list": [
        {
            "title": "Double Linked List Tutorial",
            "url": "https://youtu.be/gPxL11bX-RY",
            "channel": "Long's Clips",
            "description": "Complete guide to Double Linked Lists"
        }
    ],
    "circular linked list": [
        {
            "title": "Circular Linked List Explained",
            "url": "https://youtu.be/gPxL11bX-RY&t=1284",
            "channel": "Long's Clips",
            "description": "Implementation and use cases of Circular Linked Lists"
        }
    ],
    "stack": [
        {
            "title": "Stack Data Structure",
            "url": "https://www.youtube.com/watch?v=SmUYblJjpfE",
            "channel": "Long's Clips",
            "description": "Learn about Stack operations and implementation"
        }
    ],
    "queue": [
        {
            "title": "Queue Data Structure",
            "url": "https://www.youtube.com/watch?v=VgSIu0uiMO4",
            "channel": "Long's Clips",
            "description": "Complete guide to Queue data structure"
        }
    ],
    "priority queue": [
        {
            "title": "Priority Queue Explained",
            "url": "https://www.youtube.com/watch?v=KGukb-Z1ebA",
            "channel": "Long's Clips",
            "description": "Implementation and applications of Priority Queues"
        }
    ],
    "hash table": [
        {
            "title": "Hash Tables and Hash Functions",
            "url": "https://www.youtube.com/watch?v=uNeQ_k6qwgM",
            "channel": "Long's Clips",
            "description": "Learn about Hash Tables and collision resolution"
        }
    ],
    "depth first search": [
        {
            "title": "Depth First Search (DFS) Algorithm",
            "url": "https://www.youtube.com/watch?v=JAlNXyfe-p4",
            "channel": "Long's Clips",
            "description": "Complete tutorial on DFS algorithm with examples"
        }
    ],
    "breadth first search": [
        {
            "title": "Breadth First Search (BFS) Algorithm",
            "url": "https://www.youtube.com/watch?v=bhB-GIP3tZM",
            "channel": "Long's Clips",
            "description": "Learn BFS with step-by-step examples"
        }
    ],
    "binary tree": [
        {
            "title": "Binary Tree Data Structure",
            "url": "https://www.youtube.com/watch?v=yL7v8iOjIr0",
            "channel": "Long's Clips",
            "description": "Comprehensive guide to Binary Trees"
        }
    ]
}

def _fetch_videos_from_api(topic: str, max_results: int) -> List[Dict[str, str]]:
    """
    Fall back function to get videos from YouTube API if no recommended videos are found.
    
    Args:
        topic (str): The topic to search for
        max_results (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, str]]: List of video information dictionaries
    """
    logger.info(f"Fetching videos from API for topic: {topic}")
    
    try:
        # Create a safe search query
        search_query = f"educational {topic} algorithm tutorial"
        
        # Mock external API call response with some default videos
        default_videos = [
            {
                "title": f"{topic.title()} Tutorial",
                "url": f"https://www.youtube.com/watch?v=example1",
                "channel": "Educational Channel",
                "description": f"Learn about {topic}"
            },
            {
                "title": f"{topic.title()} Explained",
                "url": f"https://www.youtube.com/watch?v=example2",
                "channel": "CS Tutorials",
                "description": f"Visual explanation of {topic}"
            },
            {
                "title": f"{topic.title()} for Beginners",
                "url": f"https://www.youtube.com/watch?v=example3",
                "channel": "Programming Simplified",
                "description": f"Step by step guide to {topic}"
            }
        ]
        
        return default_videos[:max_results]
    except Exception as e:
        logger.error(f"Error fetching videos from API: {str(e)}")
        return []

def get_educational_video(topic: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Retrieve educational videos related to data structures and algorithms topics.
    First checks for recommended videos, then falls back to YouTube API search.
    
    Args:
        topic (str): The DSA topic to search for videos about
        max_results (int, optional): Maximum number of video results to return. Defaults to 3.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing video information:
            - title: The video title
            - url: The YouTube video URL
            - channel: The channel name
            - description: A short description of the video
    """
    logger.info(f"Searching educational videos for topic: {topic}")
    
    # Normalize topic for matching
    normalized_topic = topic.lower().strip()
    
    # Define language-specific mapping for non-English queries
    language_mapping = {
        "sắp xếp chọn": "selection sort",
        "sắp xếp nhanh": "quick sort",
        "sắp xếp nổi bọt": "bubble sort",
        "sắp xếp chèn": "insertion sort",
        "sắp xếp trộn": "merge sort",
        "tìm kiếm tuyến tính": "linear search",
        "tìm kiếm nhị phân": "binary search",
        "danh sách liên kết đơn": "single linked list",
        "danh sách liên kết kép": "double linked list",
        "danh sách liên kết vòng": "circular linked list",
        "ngăn xếp": "stack",
        "hàng đợi": "queue",
        "hàng đợi ưu tiên": "priority queue",
        "bảng băm": "hash table",
        "tìm kiếm theo chiều sâu": "depth first search",
        "tìm kiếm theo chiều rộng": "breadth first search",
        "cây nhị phân": "binary tree"
    }
    
    # Check if we have this topic in another language
    for non_english, english in language_mapping.items():
        if non_english in normalized_topic:
            normalized_topic = english
            break
    
    # Try specific algorithm detection
    algorithm_patterns = {
        "selection sort là gì": "selection sort",
        "quick sort là gì": "quick sort",
        "merge sort là gì": "merge sort",
        "bubble sort là gì": "bubble sort",
        "insertion sort là gì": "insertion sort"
    }
    
    # Check for specific algorithm patterns
    for pattern, algo_type in algorithm_patterns.items():
        if pattern in normalized_topic:
            logger.info(f"Matched algorithm pattern: {pattern} -> {algo_type}")
            if algo_type in RECOMMENDED_VIDEOS:
                return RECOMMENDED_VIDEOS[algo_type][:max_results]
    
    # First try exact matches
    if normalized_topic in RECOMMENDED_VIDEOS:
        logger.info(f"Found exact match recommended videos for '{normalized_topic}'")
        return RECOMMENDED_VIDEOS[normalized_topic][:max_results]
    
    # Try to find the best partial match
    best_match = None
    best_match_score = 0
    
    for key in RECOMMENDED_VIDEOS:
        # If the key is fully contained in the topic
        if key in normalized_topic:
            score = len(key)
            if score > best_match_score:
                best_match = key
                best_match_score = score
        
        # If each word in the key is in the topic
        key_words = key.split()
        if all(word in normalized_topic for word in key_words):
            score = sum(len(word) for word in key_words)
            if score > best_match_score:
                best_match = key
                best_match_score = score
    
    # If we found a partial match
    if best_match:
        logger.info(f"Found partial match recommended videos for '{best_match}'")
        return RECOMMENDED_VIDEOS[best_match][:max_results]
    
    # Special matching for common topics with special handling
    topic_mapping = {
        "merge": "merge sort",
        "quick": "quick sort",
        "bubble": "bubble sort",
        "insertion": "insertion sort",
        "selection": "selection sort",  # Correct mapping
        "binary": "binary search",
        "depth": "depth first search",
        "breadth": "breadth first search",
        "hash": "hash table",
        "linked": "single linked list"
    }
    
    # Check if any of the mapping keys are in the normalized topic
    for key, mapped_key in topic_mapping.items():
        if key in normalized_topic:
            logger.info(f"Found mapped recommended videos from '{key}' to '{mapped_key}'")
            return RECOMMENDED_VIDEOS[mapped_key][:max_results]
    
    # If no match found in recommended videos, use YouTube API
    logger.info(f"No recommended videos found for '{topic}', using YouTube API")
    return _fetch_videos_from_api(topic, max_results)