import re
import json
import time
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import difflib
import nltk
from nltk.tokenize import word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
    "videos": ["<VIDEOS>", "</VIDEOS>"],
    "lab": ["<LAB>", "</LAB>"] 
}

# Keywords for different DSA topics to check relevance
# Keywords for different DSA topics to check relevance
DSA_KEYWORDS = {
    "sort": ["sort", "sorting", "order", "arrange", "sequence", "sắp xếp", "xếp", "thứ tự"],
    "search": ["search", "find", "locate", "query", "lookup", "tìm kiếm", "truy vấn", "tìm", "tra cứu"],
    "tree": ["tree", "binary", "node", "leaf", "root", "branch", "cây", "nhị phân", "nút", "lá", "gốc", "nhánh"],
    "graph": ["graph", "vertex", "edge", "node", "path", "network", "đồ thị", "đỉnh", "cạnh", "nút", "đường đi", "mạng"],
    "hash": ["hash", "map", "key", "value", "collision", "bucket", "bảng băm", "khóa", "giá trị", "va chạm", "băm"],
    "queue": ["queue", "enqueue", "dequeue", "fifo", "first in first out", "hàng đợi", "vào trước ra trước"],
    "stack": ["stack", "push", "pop", "lifo", "last in first out", "ngăn xếp", "vào sau ra trước", "đẩy", "lấy"],
    "linked_list": ["linked list", "node", "pointer", "next", "previous", "danh sách liên kết", "con trỏ", "kế tiếp", "trước đó"],
    "array": ["array", "index", "element", "position", "contiguous", "mảng", "phần tử", "chỉ mục", "vị trí"],
    "recursion": ["recursion", "recursive", "base case", "call stack", "đệ quy", "trường hợp cơ sở", "trường hợp cơ bản"],
    "dynamic_programming": ["dynamic programming", "dp", "memoization", "subproblem", "quy hoạch động", "bài toán con", "ghi nhớ"],
    "greedy": ["greedy", "optimal", "choice", "local optimum", "tham lam", "tối ưu cục bộ", "lựa chọn"]
}

# Store evaluation results
EVAL_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evaluations"))
EVAL_DIR.mkdir(exist_ok=True)

class EvaluationMetrics:
    """Collection of algorithms for evaluating response quality"""
    
    @staticmethod
    def jaccard_similarity(set1, set2):
        """Calculate Jaccard similarity between two sets"""
        intersection = len(set(set1).intersection(set(set2)))
        union = len(set(set1).union(set(set2)))
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def cosine_similarity(text1, text2):
        """Simple cosine similarity between two texts"""
        # Tokenize and create word sets
        tokens1 = word_tokenize(text1.lower())
        tokens2 = word_tokenize(text2.lower())
        
        # Count word frequencies
        word_freq1 = {}
        word_freq2 = {}
        
        for word in tokens1:
            word_freq1[word] = word_freq1.get(word, 0) + 1
            
        for word in tokens2:
            word_freq2[word] = word_freq2.get(word, 0) + 1
            
        # Find all unique words
        all_words = set(word_freq1.keys()).union(set(word_freq2.keys()))
        
        # Calculate dot product
        dot_product = sum(word_freq1.get(word, 0) * word_freq2.get(word, 0) for word in all_words)
        
        # Calculate magnitudes
        magnitude1 = sum(word_freq1.get(word, 0) ** 2 for word in all_words) ** 0.5
        magnitude2 = sum(word_freq2.get(word, 0) ** 2 for word in all_words) ** 0.5
        
        # Calculate cosine similarity
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def code_syntactic_correctness(code):
        """Check for basic syntax errors in code"""
        # Check balanced braces, parentheses and brackets
        def is_balanced(code_str):
            stack = []
            for char in code_str:
                if char in '({[':
                    stack.append(char)
                elif char in ')}]':
                    if not stack:
                        return False
                    if (char == ')' and stack[-1] != '(') or \
                       (char == '}' and stack[-1] != '{') or \
                       (char == ']' and stack[-1] != '['):
                        return False
                    stack.pop()
            return len(stack) == 0
        
        # Check for common syntax patterns in C/C++
        has_semicolons = ';' in code
        has_function_definitions = bool(re.search(r'\w+\s+\w+\s*\([^)]*\)\s*\{', code))
        has_control_structures = bool(re.search(r'(if|for|while|switch)\s*\(', code))
        
        # Assign scores based on heuristics
        score = 0
        if is_balanced(code):
            score += 40
        if has_semicolons:
            score += 20
        if has_function_definitions:
            score += 20
        if has_control_structures:
            score += 20
            
        return score
    
    @staticmethod
    def quality_of_explanation(text):
        """Rate the quality of an explanation"""
        # Check for educational indicators
        edu_indicators = [
            "first", "second", "third", "next", "then", "finally",
            "because", "therefore", "thus", "as a result",
            "for example", "consider", "note that",
            "important", "key", "crucial", "essential",
            "trước tiên", "tiếp theo", "cuối cùng",
            "bởi vì", "vì vậy", "do đó", "kết quả là",
            "ví dụ", "lưu ý", "quan trọng"
        ]
        
        # Check for technical terms related to DSA
        technical_terms = [
            "algorithm", "complexity", "time", "space", 
            "O(n)", "O(log n)", "O(n²)", "O(1)",
            "worst case", "best case", "average case",
            "thuật toán", "độ phức tạp", "thời gian", "không gian"
        ]
        
        # Calculate score
        explanation_score = 0
        
        # Count educational indicators
        edu_count = sum(1 for indicator in edu_indicators if indicator.lower() in text.lower())
        explanation_score += min(40, edu_count * 8)  # Max 40 points
        
        # Count technical terms
        tech_count = sum(1 for term in technical_terms if term.lower() in text.lower())
        explanation_score += min(30, tech_count * 6)  # Max 30 points
        
        # Check for sentence structure variety (approximate)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.strip().split()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
        sentence_length_score = 0
        if 5 <= avg_sentence_length <= 15:
            sentence_length_score = 15  # Good readability
        elif 15 < avg_sentence_length <= 25:
            sentence_length_score = 10  # Acceptable but could be complex
        else:
            sentence_length_score = 5   # Too short or too complex
            
        explanation_score += sentence_length_score
        
        # Check for presence of examples
        if "for example" in text.lower() or "ví dụ" in text.lower() or "example" in text.lower():
            explanation_score += 15
        
        return min(100, explanation_score)  # Max 100
    
    @staticmethod
    def topic_relevance(query, response, topic=None):
        """Measure relevance of the response to the query topic"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Try to detect the topic if not provided
        if not topic:
            topic = None
            for potential_topic, keywords in DSA_KEYWORDS.items():
                if any(keyword in query_lower for keyword in keywords):
                    topic = potential_topic
                    break
        
        relevance_score = 0
        
        # Direct query term matching
        query_words = set(word.lower() for word in query_lower.split() if len(word) > 3)
        response_words = set(word.lower() for word in response_lower.split())
        
        query_match_ratio = len(query_words.intersection(response_words)) / max(1, len(query_words))
        relevance_score += min(40, int(query_match_ratio * 40))
        
        # Topic-specific keyword matching
        if topic and topic in DSA_KEYWORDS:
            topic_keywords = DSA_KEYWORDS[topic]
            topic_keyword_count = sum(1 for keyword in topic_keywords if keyword.lower() in response_lower)
            relevance_score += min(40, topic_keyword_count * 8)
        else:
            # Generic DSA terminology if no specific topic detected
            all_keywords = [kw for sublist in DSA_KEYWORDS.values() for kw in sublist]
            keyword_count = sum(1 for keyword in all_keywords if keyword.lower() in response_lower)
            relevance_score += min(30, keyword_count * 3)
        
        # Check for structure matching query intent
        is_conceptual_query = any(indicator in query_lower for indicator in 
                               ["what is", "explain", "concept", "define", "là gì", "khái niệm", "định nghĩa"])
        is_implementation_query = any(indicator in query_lower for indicator in 
                                  ["implement", "code", "how to", "write", "triển khai", "viết mã"])
        
        if (is_conceptual_query and "<CONCEPT>" in response) or (is_implementation_query and "<IMPLEMENTATION>" in response):
            relevance_score += 20
        elif not is_conceptual_query and not is_implementation_query:
            relevance_score += 10  # Neutral case
        
        return min(100, relevance_score)  # Max 100

def save_evaluation(query: str, evaluation: Dict) -> None:
    """
    Save evaluation results to a JSON file
    
    Args:
        query: User's query
        evaluation: Evaluation results dictionary
    """
    # Create a unique filename based on timestamp and query
    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
    filepath = EVAL_DIR / filename
    
    # Save the evaluation data to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation saved to {filepath}")
    return str(filepath)

def evaluate_response(query: str, response: str, topic: Optional[str] = None, conversation_id: Optional[str] =None) -> Dict:
    """
    Evaluate the quality of the model's response using algorithmic metrics
    
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
        "timestamp": time.time(),
        "query": query,
        "conversation_id": conversation_id or str(uuid.uuid4())
    }
    
    # 1. Check structure conformity (30% of total score)
    structure_results = evaluate_structure(response)
    evaluation["structure_score"] = structure_results["score"]
    evaluation["details"]["structure"] = structure_results
    
    # 2. Evaluate content quality (40% of total score)
    content_results = evaluate_content(response)
    evaluation["content_score"] = content_results["score"]
    evaluation["details"]["content"] = content_results
    
    # 3. Check relevance to query (30% of total score)
    relevance_results = evaluate_relevance(query, response, topic)
    evaluation["relevance_score"] = relevance_results["score"]
    evaluation["details"]["relevance"] = relevance_results
    
    # Calculate combined score (weighted average)
    evaluation["combined_score"] = (
        0.3 * evaluation["structure_score"] +
        0.4 * evaluation["content_score"] +
        0.3 * evaluation["relevance_score"]
    )
    
    # Save evaluation result
    eval_path = save_evaluation(query, evaluation)
    
    return evaluation, eval_path

def extract_tag_content(response: str, tag_name: str) -> str:
    """Extract content between opening and closing tags"""
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    
    start_pos = response.find(open_tag)
    end_pos = response.find(close_tag)
    
    if start_pos >= 0 and end_pos >= 0 and start_pos < end_pos:
        return response[start_pos + len(open_tag):end_pos].strip()
    return ""

def evaluate_structure(response: str) -> Dict:
    """Check if the response follows the expected structure with HTML tags"""
    results = {"score": 0, "findings": []}
    
    # Determine which category of response this should be
    if any(tag in response for tag in ["<CONCEPT>", "<EXAMPLE>", "<VISUALIZATION>"]):
        # This is a concept explanation
        expected_tag_types = ["concept", "example", "visualization"]
        is_concept = True
    elif any(tag in response for tag in ["<IMPLEMENTATION>", "<EXPLAINATION>", "<COMPLEXITY>"]):
        # This is an implementation response
        expected_tag_types = ["implementation", "explanation", "complexity"]
        is_concept = False
    else:
        # Can't determine the type, use all possible tags
        expected_tag_types = EXPECTED_TAGS.keys()
        is_concept = None
    
    # Check tag presence, completeness and proper nesting
    tag_score = 0
    total_expected = len(expected_tag_types)
    found_tag_types = 0
    
    for tag_type in expected_tag_types:
        open_tag, close_tag = EXPECTED_TAGS[tag_type]
        
        # Check if tags exist and are properly nested
        open_pos = response.find(open_tag)
        close_pos = response.find(close_tag)
        
        if open_pos >= 0 and close_pos >= 0:
            if open_pos < close_pos:
                # Tags exist and are properly nested
                found_tag_types += 1
                # Extract content and check if it's not empty
                content = response[open_pos + len(open_tag):close_pos].strip()
                if content:
                    results["findings"].append(f"{tag_type.capitalize()} tag present with content")
                    tag_score += 100 / total_expected
                else:
                    results["findings"].append(f"{tag_type.capitalize()} tag present but empty")
                    tag_score += 50 / total_expected
            else:
                results["findings"].append(f"{tag_type.capitalize()} tags present but improperly nested")
        elif open_pos >= 0:
            results["findings"].append(f"{tag_type.capitalize()} has opening tag but missing closing tag")
        elif close_pos >= 0:
            results["findings"].append(f"{tag_type.capitalize()} has closing tag but missing opening tag")
        else:
            results["findings"].append(f"{tag_type.capitalize()} tags missing completely")
    
    lab_open_tag, lab_close_tag = EXPECTED_TAGS["lab"]
    lab_open_pos = response.find(lab_open_tag)
    lab_close_pos = response.find(lab_close_tag)
    
    if lab_open_pos >= 0 and lab_close_pos >= 0:
        if lab_open_pos < lab_close_pos:
            # LAB tags exist and are properly nested
            lab_content = response[lab_open_pos + len(lab_open_tag):lab_close_pos].strip()
            if lab_content:
                results["findings"].append("Lab tag present with homework content")
                tag_score = min(100, tag_score * 1.1)  # Give a small bonus for including homework
            else:
                results["findings"].append("Lab tag present but empty")
        else:
            results["findings"].append("Lab tags present but improperly nested")
    elif lab_open_pos >= 0:
        results["findings"].append("Lab has opening tag but missing closing tag")
    elif lab_close_pos >= 0:
        results["findings"].append("Lab has closing tag but missing opening tag")
    # No need to report missing LAB tags as they are optional

    # Special checks for implementation responses
    if is_concept is False:  # Implementation response
        implementation_content = extract_tag_content(response, "IMPLEMENTATION")
        
        # Check for code blocks
        has_code_block = "```" in implementation_content
        if has_code_block:
            # Extract code and check syntax
            code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', implementation_content, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
                code_quality = EvaluationMetrics.code_syntactic_correctness(code)
                tag_score = tag_score * (0.7 + (0.3 * code_quality / 100))
                results["findings"].append(f"Code quality assessment: {code_quality}%")
            else:
                results["findings"].append("Code block format is incorrect")
                tag_score *= 0.8
        else:
            results["findings"].append("Implementation section should contain code blocks")
            tag_score *= 0.7
    
    # Check for special case of videos tag (optional but good to have)
    videos_content = extract_tag_content(response, "VIDEOS") 
    if videos_content:
        results["findings"].append("Videos section present")
        tag_score = min(100, tag_score * 1.1)  # Give a small bonus
    
    results["score"] = min(100, tag_score)
    
    # Add summary
    if results["score"] >= 80:
        results["summary"] = "Excellent structure with properly formatted HTML tags"
        results["summary_vi"] = "Cấu trúc xuất sắc với các thẻ HTML được định dạng đúng"
    elif results["score"] >= 60:
        results["summary"] = "Good structure with most required tags present"
        results["summary_vi"] = "Cấu trúc tốt với hầu hết các thẻ yêu cầu"
    elif results["score"] >= 40:
        results["summary"] = "Adequate structure but some tags are missing or improperly used"
        results["summary_vi"] = "Cấu trúc đầy đủ nhưng một số thẻ bị thiếu hoặc sử dụng không đúng"
    else:
        results["summary"] = "Poor structure with significant tag issues or missing sections"
        results["summary_vi"] = "Cấu trúc kém với nhiều vấn đề về thẻ hoặc thiếu các phần quan trọng"
    
        # Add Vietnamese findings
    findings_vi = []
    for finding in results["findings"]:
        if "tag present with content" in finding:
            tag_type = finding.split()[0]
            findings_vi.append(f"Thẻ {tag_type} có nội dung đầy đủ")
        elif "tag present but empty" in finding:
            tag_type = finding.split()[0]
            findings_vi.append(f"Thẻ {tag_type} hiện diện nhưng trống")
        elif "has opening tag but missing closing tag" in finding:
            tag_type = finding.split()[0]
            findings_vi.append(f"Thẻ {tag_type} có mở nhưng thiếu đóng")
        elif "has closing tag but missing opening tag" in finding:
            tag_type = finding.split()[0]
            findings_vi.append(f"Thẻ {tag_type} có đóng nhưng thiếu mở") 
        elif "tags present but improperly nested" in finding:
            tag_type = finding.split()[0]
            findings_vi.append(f"Thẻ {tag_type} hiện diện nhưng lồng không đúng")
        elif "tags missing completely" in finding:
            tag_type = finding.split()[0]
            findings_vi.append(f"Thiếu hoàn toàn thẻ {tag_type}")
        elif "Videos section present" in finding:
            findings_vi.append("Phần Videos hiện diện")
        elif "Code quality assessment" in finding:
            quality = finding.split(": ")[1]
            findings_vi.append(f"Đánh giá chất lượng mã: {quality}")
        elif "Code block format is incorrect" in finding:
            findings_vi.append("Định dạng khối mã không chính xác")
        elif "Implementation section should contain code blocks" in finding:
            findings_vi.append("Phần triển khai nên chứa các khối mã")
        else:
            # For any unmatched findings, create a generic Vietnamese translation
            findings_vi.append(f"Phát hiện: {finding}")
    
    results["findings_vi"] = findings_vi

    return results

def evaluate_content(response: str) -> Dict:
    """Evaluate content quality using algorithmic metrics"""
    results = {"score": 0, "findings": []}
    
    # Extract content from different sections
    concept_content = extract_tag_content(response, "CONCEPT")
    example_content = extract_tag_content(response, "EXAMPLE")
    implementation_content = extract_tag_content(response, "IMPLEMENTATION")
    explanation_content = extract_tag_content(response, "EXPLAINATION")
    complexity_content = extract_tag_content(response, "COMPLEXITY")
    
    # Determine which type of response we're evaluating
    is_concept_response = len(concept_content) > 0 or len(example_content) > 0
    is_implementation_response = len(implementation_content) > 0 or len(explanation_content) > 0
    
    content_score = 0
    
    # 1. Content quality analysis
    if is_concept_response:
        # Evaluate conceptual explanation
        explanation_quality = EvaluationMetrics.quality_of_explanation(concept_content)
        results["findings"].append(f"Concept explanation quality: {explanation_quality}%")
        content_score += explanation_quality * 0.5
        
        # Evaluate examples
        if example_content:
            example_quality = EvaluationMetrics.quality_of_explanation(example_content)
            results["findings"].append(f"Example quality: {example_quality}%")
            content_score += example_quality * 0.3
        else:
            results["findings"].append("Missing examples for the concept")
            
        # Check for visualization description
        visualization_content = extract_tag_content(response, "VISUALIZATION")
        if visualization_content:
            if len(visualization_content) > 100:
                results["findings"].append("Good visualization description")
                content_score += 20
            else:
                results["findings"].append("Brief visualization description")
                content_score += 10
        else:
            results["findings"].append("Missing visualization description")
            
    elif is_implementation_response:
        # Evaluate code quality
        if implementation_content:
            # Extract code blocks
            code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', implementation_content, re.DOTALL)
            if code_blocks:
                code = code_blocks[0]
                
                # Check for comments
                comment_lines = len(re.findall(r'//.*|/\*.*?\*/|#.*', code))
                code_lines = len(code.strip().split('\n'))
                comment_ratio = comment_lines / max(1, code_lines)
                
                if comment_ratio >= 0.2:
                    results["findings"].append("Good code comments coverage")
                    content_score += 25
                elif comment_ratio > 0:
                    results["findings"].append("Some code comments present")
                    content_score += 15
                else:
                    results["findings"].append("Code lacks comments")
            else:
                results["findings"].append("Implementation doesn't contain proper code blocks")
        else:
            results["findings"].append("Missing implementation code")
            
        # Evaluate explanation quality
        if explanation_content:
            explanation_quality = EvaluationMetrics.quality_of_explanation(explanation_content)
            results["findings"].append(f"Code explanation quality: {explanation_quality}%")
            content_score += explanation_quality * 0.4
        else:
            results["findings"].append("Missing explanation for the implementation")
            
        # Check complexity analysis
        if complexity_content:
            has_big_o = bool(re.search(r'O\([^)]+\)', complexity_content))
            mentions_time_space = "time" in complexity_content.lower() and "space" in complexity_content.lower()
            
            if has_big_o and mentions_time_space:
                results["findings"].append("Complete complexity analysis with Big O notation")
                content_score += 25
            elif has_big_o or mentions_time_space:
                results["findings"].append("Partial complexity analysis")
                content_score += 15
            else:
                results["findings"].append("Basic complexity information")
                content_score += 5
        else:
            results["findings"].append("Missing complexity analysis")
    
    # Adjust for general content quality factors
    # Calculate approximate word count
    word_count = len(response.split())
    
    if word_count >= 500:
        results["findings"].append("Comprehensive response length")
        content_score = min(100, content_score * 1.1)
    elif word_count >= 300:
        results["findings"].append("Good response length")
    elif word_count < 150:
        results["findings"].append("Response is quite short")
        content_score *= 0.8
    
    # Check for educational language indicators
    educational_indicators = ["note", "important", "remember", "key point", "lưu ý", "quan trọng", "nhớ rằng"]
    if any(indicator in response.lower() for indicator in educational_indicators):
        results["findings"].append("Contains educational emphasis elements")
        content_score = min(100, content_score * 1.05)
    
    # Final content score
    results["score"] = min(100, content_score)

        # Add Vietnamese findings
    findings_vi = []
    for finding in results["findings"]:
        if "Concept explanation quality" in finding:
            quality = finding.split(": ")[1]
            findings_vi.append(f"Chất lượng giải thích khái niệm: {quality}")
        elif "Example quality" in finding:
            quality = finding.split(": ")[1]
            findings_vi.append(f"Chất lượng ví dụ: {quality}")
        elif "Good visualization description" in finding:
            findings_vi.append("Mô tả trực quan tốt")
        elif "Brief visualization description" in finding:
            findings_vi.append("Mô tả trực quan ngắn gọn")
        elif "Missing visualization description" in finding:
            findings_vi.append("Thiếu mô tả trực quan")
        elif "Good code comments coverage" in finding:
            findings_vi.append("Bao phủ chú thích mã tốt")
        elif "Some code comments present" in finding:
            findings_vi.append("Có một số chú thích mã")
        elif "Code lacks comments" in finding:
            findings_vi.append("Mã thiếu chú thích")
        elif "Code explanation quality" in finding:
            quality = finding.split(": ")[1]
            findings_vi.append(f"Chất lượng giải thích mã: {quality}")
        elif "Complete complexity analysis" in finding:
            findings_vi.append("Phân tích độ phức tạp đầy đủ với ký hiệu Big O")
        elif "Partial complexity analysis" in finding:
            findings_vi.append("Phân tích độ phức tạp một phần")
        elif "Basic complexity information" in finding:
            findings_vi.append("Thông tin độ phức tạp cơ bản")
        elif "Implementation doesn't contain proper code blocks" in finding:
            findings_vi.append("Triển khai không chứa các khối mã phù hợp")
        elif "Missing implementation code" in finding:
            findings_vi.append("Thiếu mã triển khai")
        elif "Missing explanation for the implementation" in finding:
            findings_vi.append("Thiếu giải thích cho phần triển khai")
        elif "Missing complexity analysis" in finding:
            findings_vi.append("Thiếu phân tích độ phức tạp")
        elif "Comprehensive response length" in finding:
            findings_vi.append("Độ dài câu trả lời toàn diện")
        elif "Good response length" in finding:
            findings_vi.append("Độ dài câu trả lời tốt")
        elif "Response is quite short" in finding:
            findings_vi.append("Câu trả lời khá ngắn")
        elif "Contains educational emphasis elements" in finding:
            findings_vi.append("Chứa các yếu tố nhấn mạnh giáo dục")
        elif "Missing examples for the concept" in finding:
            findings_vi.append("Thiếu ví dụ cho khái niệm")
        else:
            # For any unmatched findings, create a generic Vietnamese translation
            findings_vi.append(f"Phát hiện: {finding}")
    # Extract lab content
    lab_content = extract_tag_content(response, "LAB")
    
    # Add evaluation for LAB content if present
    if lab_content:
        # Check lab quality based on length and structure
        lab_quality = EvaluationMetrics.quality_of_explanation(lab_content)
        
        # Check if it contains actual exercises/tasks
        has_exercises = any(kw in lab_content.lower() for kw in 
                         ["exercise", "task", "implement", "write", "create", "practice",
                          "bài tập", "thực hành", "triển khai", "viết", "tạo", "thực hiện"])
        
        if has_exercises:
            results["findings"].append(f"Lab content contains clear exercises: {lab_quality}%")
            content_score = min(100, content_score * 1.05)  # Small bonus for good homework
        else:
            results["findings"].append("Lab content present but lacks clear exercises")
    
    # In evaluate_structure, add to the findings_vi section:
    elif "Lab tag present with homework content" in finding:
        findings_vi.append("Thẻ Lab có nội dung bài tập")
    elif "Lab tag present but empty" in finding:
        findings_vi.append("Thẻ Lab hiện diện nhưng trống")
    elif "Lab tags present but improperly nested" in finding:
        findings_vi.append("Thẻ Lab hiện diện nhưng lồng không đúng")
    elif "Lab has opening tag but missing closing tag" in finding:
        findings_vi.append("Thẻ Lab có mở nhưng thiếu đóng")
    elif "Lab has closing tag but missing opening tag" in finding:
        findings_vi.append("Thẻ Lab có đóng nhưng thiếu mở")

    # In evaluate_content, add to the findings_vi section:
    elif "Lab content contains clear exercises" in finding:
        quality = finding.split(": ")[1]
        findings_vi.append(f"Nội dung bài tập Lab rõ ràng: {quality}")
    elif "Lab content present but lacks clear exercises" in finding:
        findings_vi.append("Nội dung Lab hiện diện nhưng thiếu bài tập rõ ràng")
    
    results["findings_vi"] = findings_vi
    
    # Add summary
    if results["score"] >= 80:
        results["summary"] = "Excellent content with comprehensive explanations and examples"
        results["summary_vi"] = "Nội dung xuất sắc với các giải thích và ví dụ toàn diện"
    elif results["score"] >= 60:
        results["summary"] = "Good content with solid explanations"
        results["summary_vi"] = "Nội dung tốt với các giải thích vững chắc"
    elif results["score"] >= 40:
        results["summary"] = "Adequate content but could be improved in depth or clarity"
        results["summary_vi"] = "Nội dung đầy đủ nhưng có thể cải thiện về độ sâu hoặc độ rõ ràng"
    else:
        results["summary"] = "Poor content quality with significant gaps or issues"
        results["summary_vi"] = "Chất lượng nội dung kém với những thiếu sót hoặc vấn đề đáng kể"
    
    return results

def evaluate_relevance(query: str, response: str, topic: Optional[str] = None) -> Dict:
    """Evaluate relevance of the response to the query"""
    results = {"score": 0, "findings": []}
    
    # Use the topic relevance algorithm
    relevance_score = EvaluationMetrics.topic_relevance(query, response, topic)
    results["score"] = relevance_score
    
    # Additional relevance checks
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Check for direct word matches between query and response
    # excluding common words
    common_words = ["the", "a", "an", "is", "are", "in", "on", "at", "to", "and", "or", "of", 
                   "with", "by", "for", "about", "các", "những", "là", "và", "hoặc", "về"]
    
    query_important_words = [word for word in query_lower.split() 
                           if len(word) > 3 and word not in common_words]
    
    matched_words = [word for word in query_important_words if word in response_lower]
    match_percentage = len(matched_words) / max(1, len(query_important_words)) * 100
    
    if match_percentage >= 75:
        results["findings"].append("Response directly addresses most query terms")
    elif match_percentage >= 50:
        results["findings"].append("Response addresses many query terms")
    elif match_percentage >= 25:
        results["findings"].append("Response addresses some query terms")
    else:
        results["findings"].append("Response addresses few direct query terms")
        results["score"] = max(10, results["score"] * 0.8)  # Penalize for low direct relevance

    # Check if the response matches the query intent
    is_conceptual_query = any(term in query_lower for term in 
                           ["what", "explain", "concept", "mean", "definition", "define", "là gì", "khái niệm"])
    has_conceptual_response = "<CONCEPT>" in response
    
    is_implementation_query = any(term in query_lower for term in 
                               ["how", "implement", "code", "program", "write", "triển khai", "viết", "mã"])
    has_implementation_response = "<IMPLEMENTATION>" in response
    
    # Adjust score based on intent matching
    if (is_conceptual_query and has_conceptual_response) or (is_implementation_query and has_implementation_response):
        results["findings"].append("Response format matches query intent")
        results["score"] = min(100, results["score"] * 1.1)  # Slight boost
    elif (is_conceptual_query and not has_conceptual_response) or (is_implementation_query and not has_implementation_response):
        results["findings"].append("Response format doesn't match query intent")
        results["score"] = max(10, results["score"] * 0.9)  # Slight penalty
    
            # Add Vietnamese findings
    findings_vi = []
    for finding in results["findings"]:
        if "Response directly addresses most query terms" in finding:
            findings_vi.append("Câu trả lời trực tiếp đề cập đến hầu hết các thuật ngữ trong truy vấn")
        elif "Response addresses many query terms" in finding:
            findings_vi.append("Câu trả lời đề cập đến nhiều thuật ngữ trong truy vấn")
        elif "Response addresses some query terms" in finding:
            findings_vi.append("Câu trả lời đề cập đến một số thuật ngữ trong truy vấn")
        elif "Response addresses few direct query terms" in finding:
            findings_vi.append("Câu trả lời đề cập đến ít thuật ngữ trực tiếp trong truy vấn")
        elif "Response format matches query intent" in finding:
            findings_vi.append("Định dạng phản hồi phù hợp với mục đích truy vấn")
        elif "Response format doesn't match query intent" in finding:
            findings_vi.append("Định dạng phản hồi không phù hợp với mục đích truy vấn")
        else:
            findings_vi.append(finding)
    
    results["findings_vi"] = findings_vi

    # Add summary based on score
    if results["score"] >= 80:
        results["summary"] = "Excellent relevance to the query"
        results["summary_vi"] = "Độ liên quan xuất sắc với truy vấn"
    elif results["score"] >= 60:
        results["summary"] = "Good relevance to the query"
        results["summary_vi"] = "Độ liên quan tốt với truy vấn"
    elif results["score"] >= 40:
        results["summary"] = "Adequate relevance but could be more targeted"
        results["summary_vi"] = "Độ liên quan đầy đủ nhưng có thể tập trung hơn"
    else:
        results["summary"] = "Poor relevance to the original query"
        results["summary_vi"] = "Độ liên quan kém với truy vấn ban đầu"
    
    return results

def prompt_based_evaluation(query: str, expected_output: str, completion: str) -> Dict:
    """
    Perform prompt-based evaluation using a structured evaluator prompt
    
    Args:
        query: The user's original prompt/question
        expected_output: The expected/reference output
        completion: The generated completion to evaluate
        
    Returns:
        Dict containing evaluation results with score and explanation
    """
    from app.core.config import EVALUATION_PROMPT
    import json
    from app.core.llm import get_llm_response
    
    # Construct the evaluation prompt
    evaluation_input = EVALUATION_PROMPT.format(
        query=query,
        expected_output=expected_output,
        completion=completion
    )
    
    # Get evaluation from the LLM
    try:
        raw_evaluation = get_llm_response(evaluation_input)
        
        # Extract the JSON part of the response
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_evaluation, re.DOTALL)
        if json_match:
            evaluation_json = json_match.group(1)
            evaluation_results = json.loads(evaluation_json)
        else:
            # Try to find any JSON-like structure in the response
            json_like_match = re.search(r'(\{.*\})', raw_evaluation, re.DOTALL)
            if json_like_match:
                evaluation_json = json_like_match.group(1)
                evaluation_results = json.loads(evaluation_json)
            else:
                # If all else fails, create a basic structure
                evaluation_results = {
                    "scores": {
                        "overall": 0
                    },
                    "feedback": {
                        "improvement_suggestions": "Failed to parse evaluation response"
                    }
                }
        
        # Return the structured evaluation
        return {
            "score": evaluation_results.get("scores", {}).get("overall", 0),
            "structure_score": evaluation_results.get("scores", {}).get("structure", 0),
            "technical_accuracy": evaluation_results.get("scores", {}).get("technical_accuracy", 0),
            "educational_value": evaluation_results.get("scores", {}).get("educational_value", 0),
            "completeness": evaluation_results.get("scores", {}).get("completeness", 0),
            "feedback": evaluation_results.get("feedback", {}),
            "raw_evaluation": raw_evaluation
        }
    
    except Exception as e:
        logger.error(f"Error in prompt-based evaluation: {str(e)}")
        return {
            "score": 0,
            "error": str(e),
            "raw_evaluation": "Evaluation failed due to error"
        }

def compare_outputs(expected_output: str, completion: str) -> Dict:
    """
    Compare expected and actual outputs using algorithmic methods
    
    Args:
        expected_output: The reference/expected output
        completion: The generated completion to evaluate
        
    Returns:
        Dict with similarity scores and analysis
    """
    results = {}
    
    # String similarity using difflib
    similarity_ratio = difflib.SequenceMatcher(None, expected_output, completion).ratio()
    results["string_similarity"] = similarity_ratio * 100
    
    # Word-level Jaccard similarity
    expected_words = set(word_tokenize(expected_output.lower()))
    completion_words = set(word_tokenize(completion.lower()))
    results["word_similarity"] = EvaluationMetrics.jaccard_similarity(expected_words, completion_words) * 100
    
    # Cosine similarity
    results["cosine_similarity"] = EvaluationMetrics.cosine_similarity(expected_output, completion) * 100
    
    # Code similarity check - if both contain code blocks
    expected_code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', expected_output, re.DOTALL)
    completion_code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', completion, re.DOTALL)
    
    if expected_code_blocks and completion_code_blocks:
        # Compare the first code blocks in each
        code_similarity = difflib.SequenceMatcher(None, expected_code_blocks[0], completion_code_blocks[0]).ratio()
        results["code_similarity"] = code_similarity * 100
    else:
        results["code_similarity"] = 0.0
    
    # Calculate combined score weighted by importance
    results["combined_score"] = (
        0.3 * results["string_similarity"] +
        0.3 * results["word_similarity"] +
        0.3 * results["cosine_similarity"] +
        0.1 * results.get("code_similarity", 0)
    )
    
    return results

def comprehensive_evaluation(query: str, response: str, expected_output: Optional[str] = None, topic: Optional[str] = None) -> Dict:
    """
    Perform a comprehensive evaluation using both algorithmic metrics and prompt-based methods
    
    Args:
        query: User's original query
        response: Generated response to evaluate
        expected_output: Optional reference/expected output
        topic: Optional topic classification
        
    Returns:
        Dict with combined evaluation results
    """
    results = {}
    
    # 1. Algorithmic evaluation using our metrics
    algo_eval = evaluate_response(query, response, topic)
    results["algorithmic"] = algo_eval
    
    # 2. If an expected output is provided, compare with it
    if expected_output:
        comparison = compare_outputs(expected_output, response)
        results["comparison"] = comparison
        
        # 3. Use prompt-based evaluation as well
        prompt_eval = prompt_based_evaluation(query, expected_output, response)
        results["prompt_based"] = prompt_eval
        
        # 4. Calculate an overall combined score from all methods
        results["overall_score"] = (
            0.4 * algo_eval["combined_score"] + 
            0.3 * comparison["combined_score"] + 
            0.3 * prompt_eval["score"]
        )
    else:
        # Without expected output, rely solely on algorithmic evaluation
        results["overall_score"] = algo_eval["combined_score"]
    
    # Add timestamp and metadata
    results["timestamp"] = time.time()
    results["query"] = query
    
    # Save the comprehensive evaluation
    save_comprehensive_evaluation(query, results)
    
    return results

def save_comprehensive_evaluation(query: str, evaluation: Dict) -> None:
    """
    Save comprehensive evaluation results to a JSON file
    
    Args:
        query: User's query
        evaluation: Comprehensive evaluation results dictionary
    """
    # Create a unique filename based on timestamp and query
    filename = f"comprehensive_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
    filepath = EVAL_DIR / filename
    
    # Save the evaluation data to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Comprehensive evaluation saved to {filepath}")