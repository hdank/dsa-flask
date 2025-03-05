# app/core/evaluation.py
import re
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define expected response structures
EXPECTED_TAGS = {
    "concept": ["<CONCEPT>", "</CONCEPT>"],
    "example": ["<EXAMPLE>", "</EXAMPLE>"],
    "visualization": ["<VISUALIZATION>", "</VISUALIZATION>"],
    "implementation": ["<IMPLEMENTATION>", "</IMPLEMENTATION>"],
    "explanation": ["<EXPLAINATION>", "</EXPLAINATION>"],
    "complexity": ["<COMPLEXITY>", "</COMPLEXITY>"],
    "videos": ["<VIDEOS>", "</VIDEOS>"]
}

# Keywords for different DSA topics to check relevance
DSA_KEYWORDS = {
    "sort": ["sort", "sorting", "order", "arrange", "sequence", "sắp xếp", "xếp", "thứ tự"],
    "search": ["search", "find", "locate", "query", "lookup", "tìm kiếm", "truy vấn"],
    "tree": ["tree", "binary", "node", "leaf", "root", "branch", "cây", "nhị phân", "nút", "lá", "gốc"],
    "graph": ["graph", "vertex", "edge", "node", "path", "network", "đồ thị", "đỉnh", "cạnh", "đường đi"],
    "hash": ["hash", "map", "key", "value", "collision", "bucket", "bảng băm", "khóa", "giá trị", "va chạm"],
    "queue": ["queue", "enqueue", "dequeue", "fifo", "first in first out", "hàng đợi"],
    "stack": ["stack", "push", "pop", "lifo", "last in first out", "ngăn xếp", "đẩy", "lấy"],
    "linked_list": ["linked list", "node", "pointer", "next", "previous", "danh sách liên kết", "con trỏ", "kế tiếp"],
    "array": ["array", "index", "element", "position", "contiguous", "mảng", "phần tử", "chỉ mục", "vị trí"],
    "recursion": ["recursion", "recursive", "base case", "call stack", "đệ quy", "trường hợp cơ sở"],
    "dynamic_programming": ["dynamic programming", "dp", "memoization", "subproblem", "quy hoạch động", "bài toán con"],
    "greedy": ["greedy", "optimal", "choice", "local optimum", "tham lam", "tối ưu cục bộ"]
}

# Store evaluation results
EVAL_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evaluations"))
EVAL_DIR.mkdir(exist_ok=True)

def evaluate_response(query: str, response: str, topic: Optional[str] = None) -> Dict:
    """
    Evaluate the quality of the model's response
    
    Args:
        query: User's query
        response: Model's response
        topic: Optional topic classification
        
    Returns:
        Dict containing evaluation metrics
    """
    # Initialize evaluation scores
    evaluation = {
        "structure_score": 0,
        "content_score": 0, 
        "relevance_score": 0,
        "combined_score": 0,
        "details": {},
        "timestamp": import_time(),
        "query": query,
    }
    
    # 1. Check structure conformity (40% of total score)
    structure_results = evaluate_structure(response)
    evaluation["structure_score"] = structure_results["score"]
    evaluation["details"]["structure"] = structure_results
    
    # 2. Evaluate content quality (30% of total score)
    content_results = evaluate_content(response)
    evaluation["content_score"] = content_results["score"]
    evaluation["details"]["content"] = content_results
    
    # 3. Check relevance to query (30% of total score)
    relevance_results = evaluate_relevance(query, response, topic)
    evaluation["relevance_score"] = relevance_results["score"]
    evaluation["details"]["relevance"] = relevance_results
    
    # Calculate combined score (weighted average)
    evaluation["combined_score"] = (
        0.4 * evaluation["structure_score"] +
        0.3 * evaluation["content_score"] +
        0.3 * evaluation["relevance_score"]
    )
    
    # Save evaluation result
    # save_evaluation(query, evaluation)
    
    return evaluation

def evaluate_structure(response: str) -> Dict:
    """Check if the response follows the expected structure with HTML tags"""
    results = {"score": 0, "findings": []}
    
    # Count how many expected tag pairs are present
    found_tags = 0
    total_expected_tags = 0
    
    # Determine which category of response this should be
    if any(tag in response for tag in ["<CONCEPT>", "<EXAMPLE>", "<VISUALIZATION>"]):
        # This is a concept explanation
        expected_tag_types = ["concept", "example", "visualization"]
        total_expected_tags = 3
    elif any(tag in response for tag in ["<IMPLEMENTATION>", "<EXPLAINATION>", "<COMPLEXITY>"]):
        # This is an implementation response
        expected_tag_types = ["implementation", "explanation", "complexity"]
        total_expected_tags = 3
    else:
        # Can't determine the type, use all possible tags
        expected_tag_types = EXPECTED_TAGS.keys()
        # We'll expect at least 3 tag pairs
        total_expected_tags = 3
    
    # Check for each expected tag
    for tag_type in expected_tag_types:
        open_tag, close_tag = EXPECTED_TAGS[tag_type]
        if open_tag in response and close_tag in response:
            found_tags += 1
            # Check if tags are properly nested
            open_pos = response.find(open_tag)
            close_pos = response.find(close_tag)
            if open_pos > close_pos:
                results["findings"].append(f"Tags {open_tag} and {close_tag} are not properly nested")
        else:
            if open_tag in response:
                results["findings"].append(f"Missing closing tag {close_tag}")
            elif close_tag in response:
                results["findings"].append(f"Missing opening tag {open_tag}")
            else:
                results["findings"].append(f"Missing tag pair {open_tag}...{close_tag}")
    
    # Special handling for videos section (optional)
    if "<VIDEOS>" in response and "</VIDEOS>" not in response:
        results["findings"].append("Missing closing </VIDEOS> tag")
    
    # Calculate tag structure score
    if total_expected_tags > 0:
        tag_score = min(1.0, found_tags / total_expected_tags) * 100
    else:
        tag_score = 0
        
    # Check for code blocks within implementation sections
    if "<IMPLEMENTATION>" in response and "</IMPLEMENTATION>" in response:
        implementation_content = extract_tag_content(response, "IMPLEMENTATION")
        has_code_block = "```" in implementation_content
        
        if not has_code_block:
            results["findings"].append("Implementation section lacks code block")
            tag_score -= 20  # Penalize for missing code block
        
    results["score"] = max(0, min(100, tag_score))
    
    if results["score"] >= 80:
        results["summary"] = "Good structure with proper HTML tags"
    elif results["score"] >= 50:
        results["summary"] = "Adequate structure but some tags are missing or improperly used"
    else:
        results["summary"] = "Poor structure with missing or incorrect HTML tags"
        
    return results

def evaluate_content(response: str) -> Dict:
    """Evaluate content quality based on length, format, and content indicators"""
    results = {"score": 0, "findings": []}
    score = 0
    
    # 1. Length check (20 points max)
    min_length = 100  # Characters
    ideal_length = 500
    length = len(response)
    
    if length < min_length:
        results["findings"].append(f"Response is too short ({length} chars)")
        length_score = 0
    elif length < ideal_length:
        length_score = 20 * (length - min_length) / (ideal_length - min_length)
        results["findings"].append(f"Response length is adequate ({length} chars)")
    else:
        length_score = 20
        results["findings"].append(f"Response has good length ({length} chars)")
    
    score += length_score
    
    # 2. Code quality check if code is present (30 points max)
    code_score = 0
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', response, re.DOTALL)
    
    if code_blocks:
        code = '\n'.join(code_blocks)
        
        # Check for comments
        has_comments = '//' in code or '/*' in code or '#' in code
        if has_comments:
            code_score += 10
            results["findings"].append("Code contains comments")
        else:
            results["findings"].append("Code lacks comments")
        
        # Check for proper indentation
        lines = code.split('\n')
        indented_lines = [line for line in lines if line.strip() and line[0] in [' ', '\t']]
        if len(indented_lines) > 0:
            code_score += 10
            results["findings"].append("Code has proper indentation")
        else:
            results["findings"].append("Code may lack proper indentation")
            
        # Check if code has a reasonable number of lines
        if len(lines) >= 5:
            code_score += 10
            results["findings"].append(f"Code has reasonable length ({len(lines)} lines)")
        else:
            results["findings"].append(f"Code might be too brief ({len(lines)} lines)")
    else:
        # If this is an implementation response, penalize for no code
        if "<IMPLEMENTATION>" in response:
            results["findings"].append("Implementation section should contain code blocks")
        
    score += code_score
    
    # 3. Content richness check (30 points max)
    content_richness_score = 0
    
    # Check for technical terms
    tech_terms = ["algorithm", "complexity", "time", "space", "O(", "implementation", 
                  "data structure", "function", "method", "operation"]
    
    term_count = sum(1 for term in tech_terms if term.lower() in response.lower())
    
    if term_count >= 5:
        content_richness_score += 15
        results["findings"].append(f"Good use of technical terms ({term_count} terms)")
    elif term_count >= 2:
        content_richness_score += 10
        results["findings"].append(f"Some technical terms used ({term_count} terms)")
    else:
        results["findings"].append("Few technical terms used")
    
    # Check for educational content
    edu_indicators = ["example", "consider", "note that", "important", "remember", "key", "fundamental"]
    edu_count = sum(1 for term in edu_indicators if term.lower() in response.lower())
    
    if edu_count >= 3:
        content_richness_score += 15
        results["findings"].append("Response has good educational content")
    elif edu_count >= 1:
        content_richness_score += 5
        results["findings"].append("Response has some educational elements")
    
    score += content_richness_score
    
    # 4. Check for language quality (20 points max)
    lang_score = 0
    
    # Look for signs of good explanation
    explanation_patterns = [
        r"because", r"therefore", r"hence", r"thus", r"as a result",
        r"first", r"second", r"third", r"finally", r"lastly",
        r"in other words", r"specifically", r"for example"
    ]
    
    explanation_count = sum(1 for pattern in explanation_patterns 
                           if re.search(pattern, response, re.IGNORECASE))
    
    if explanation_count >= 3:
        lang_score += 20
        results["findings"].append("Response uses good explanatory language")
    elif explanation_count >= 1:
        lang_score += 10
        results["findings"].append("Response uses some explanatory language")
    else:
        results["findings"].append("Response could use more explanatory language")
    
    score += lang_score
    
    # Final score and summary
    results["score"] = max(0, min(100, score))
    
    if results["score"] >= 80:
        results["summary"] = "High quality content with good explanations and examples"
    elif results["score"] >= 50:
        results["summary"] = "Adequate content but could be improved"
    else:
        results["summary"] = "Poor content quality, needs significant improvement"
    
    return results

def evaluate_relevance(query: str, response: str, topic: Optional[str] = None) -> Dict:
    """Evaluate relevance of the response to the query"""
    results = {"score": 0, "findings": []}
    
    query_lower = query.lower()
    response_lower = response.lower()
    
    # 1. Detect query type (40 points max)
    is_concept_query = any(concept_indicator in query_lower for concept_indicator in 
                          ["là gì", "what is", "define", "explain", "concept of", "tell me about"])
    
    is_implementation_query = any(impl_indicator in query_lower for impl_indicator in 
                                ["implement", "code", "write", "how to create", "program", "triển khai", "xây dựng"])
    
    # Check if response structure matches query type
    query_type_score = 0
    if is_concept_query and "<CONCEPT>" in response:
        query_type_score = 40
        results["findings"].append("Response structure matches concept query")
    elif is_implementation_query and "<IMPLEMENTATION>" in response:
        query_type_score = 40
        results["findings"].append("Response structure matches implementation query")
    elif not is_concept_query and not is_implementation_query:
        # Can't determine expected structure, give benefit of doubt
        query_type_score = 30
        results["findings"].append("Query type unclear, structure acceptable")
    else:
        # Structure doesn't match query type
        query_type_score = 10
        results["findings"].append("Response structure doesn't match query type")
    
    # 2. Topic relevance check (60 points max)
    topic_score = 0
    
    # Extract main topic from query if not provided
    if not topic:
        topic = detect_topic(query_lower)
    
    if topic:
        # Check if response contains topic-related keywords
        keywords = DSA_KEYWORDS.get(topic, [])
        if keywords:
            # Count keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in response_lower)
            
            if keyword_matches >= 3:
                topic_score = 60
                results["findings"].append(f"Response is highly relevant to {topic}")
            elif keyword_matches >= 1:
                topic_score = 40
                results["findings"].append(f"Response is somewhat relevant to {topic}")
            else:
                results["findings"].append(f"Response may not be relevant to {topic}")
        else:
            # No keywords for this topic
            topic_score = 30  # Give benefit of doubt
            results["findings"].append("Can't determine relevance to specific topic")
    else:
        # Couldn't detect a specific topic
        # Check if query terms appear in response
        query_terms = query_lower.split()
        query_terms = [term for term in query_terms if len(term) > 3]  # Ignore short terms
        
        if query_terms:
            term_matches = sum(1 for term in query_terms if term in response_lower)
            match_ratio = term_matches / len(query_terms)
            
            topic_score = min(60, int(match_ratio * 60))
            if topic_score >= 40:
                results["findings"].append("Response contains many query terms")
            elif topic_score >= 20:
                results["findings"].append("Response contains some query terms")
            else:
                results["findings"].append("Response contains few query terms")
        else:
            topic_score = 30  # Give benefit of doubt
            results["findings"].append("Query lacks specific terms to match")
    
    # Calculate final relevance score
    relevance_score = query_type_score + topic_score
    results["score"] = min(100, relevance_score)
    
    if results["score"] >= 80:
        results["summary"] = "Response is highly relevant to the query"
    elif results["score"] >= 50:
        results["summary"] = "Response is adequately relevant to the query"
    else:
        results["summary"] = "Response lacks relevance to the query"
    
    return results

def extract_tag_content(text: str, tag_name: str) -> str:
    """Extract content between opening and closing HTML-style tags"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def detect_topic(query: str) -> Optional[str]:
    """Detect DSA topic from query"""
    for topic, keywords in DSA_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            return topic
    return None

def import_time():
    """Get current time for timestamp"""
    import time
    return time.time()

def save_evaluation(query: str, evaluation: Dict):
    """Save evaluation results to file"""
    try:
        # Create a unique filename based on timestamp
        filename = f"eval_{int(import_time())}.json"
        filepath = EVAL_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved evaluation to {filepath}")
    except Exception as e:
        logger.error(f"Error saving evaluation: {e}")

def get_evaluation_stats(limit: int = 100) -> Dict:
    """Get statistics from recent evaluations"""
    stats = {
        "count": 0,
        "avg_structure_score": 0,
        "avg_content_score": 0,
        "avg_relevance_score": 0,
        "avg_combined_score": 0,
        "topics": {}
    }
    
    try:
        files = sorted(EVAL_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)[:limit]
        
        if not files:
            return stats
            
        total_structure = 0
        total_content = 0
        total_relevance = 0
        total_combined = 0
        topics = {}
        
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
                
                total_structure += eval_data.get("structure_score", 0)
                total_content += eval_data.get("content_score", 0)
                total_relevance += eval_data.get("relevance_score", 0)
                total_combined += eval_data.get("combined_score", 0)
                
                # Try to detect topic
                query = eval_data.get("query", "")
                topic = detect_topic(query.lower()) or "unknown"
                
                if topic in topics:
                    topics[topic] += 1
                else:
                    topics[topic] = 1
        
        count = len(files)
        stats["count"] = count
        stats["avg_structure_score"] = total_structure / count
        stats["avg_content_score"] = total_content / count
        stats["avg_relevance_score"] = total_relevance / count
        stats["avg_combined_score"] = total_combined / count
        stats["topics"] = topics
        
    except Exception as e:
        logger.error(f"Error getting evaluation stats: {e}")
        
    return stats