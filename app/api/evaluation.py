from flask import jsonify
from app.core.conversation import get_conversation_history, list_conversations

def get_eval_stats():
    """API endpoint to get evaluation statistics across all conversations"""
    # Get list of all conversations
    conversation_ids = list_conversations()
    
    # Initialize stats counters
    stats = {
        "total_messages": 0,
        "evaluated_messages": 0,
        "avg_scores": {
            "structure": 0,
            "content": 0, 
            "relevance": 0,
            "combined": 0
        },
        "score_distribution": {
            "excellent": 0,  # 80-100
            "good": 0,       # 65-80
            "fair": 0,       # 50-65
            "poor": 0        # 0-50
        },
        "topics": {}
    }
    
    # Process all conversations
    structure_total = 0
    content_total = 0
    relevance_total = 0
    combined_total = 0
    
    for conv_id in conversation_ids:
        conv_data = get_conversation_history(conv_id)
        if not conv_data or "messages" not in conv_data:
            continue
            
        messages = conv_data.get("messages", [])
        stats["total_messages"] += len([m for m in messages if m.get("role") == "assistant"])
        
        for msg in messages:
            if msg.get("role") != "assistant" or "evaluation" not in msg:
                continue
                
            stats["evaluated_messages"] += 1
            eval_data = msg["evaluation"]
            scores = eval_data.get("scores", {})
            
            # Add to totals
            structure_score = scores.get("structure", 0)
            content_score = scores.get("content", 0)
            relevance_score = scores.get("relevance", 0)
            combined_score = scores.get("combined", 0)
            
            structure_total += structure_score
            content_total += content_score
            relevance_total += relevance_score
            combined_total += combined_score
            
            # Update score distribution
            if combined_score >= 80:
                stats["score_distribution"]["excellent"] += 1
            elif combined_score >= 65:
                stats["score_distribution"]["good"] += 1
            elif combined_score >= 50:
                stats["score_distribution"]["fair"] += 1
            else:
                stats["score_distribution"]["poor"] += 1
                
            # Topic analysis - extract from user query
            if msg.get("evaluation", {}).get("query"):
                query = msg["evaluation"]["query"].lower()
                topic = None
                
                # Simple topic detection
                for potential_topic in ["sort", "search", "tree", "graph", "hash", 
                                       "queue", "stack", "linked_list", "array"]:
                    if potential_topic in query:
                        topic = potential_topic
                        break
                        
                if topic:
                    if topic not in stats["topics"]:
                        stats["topics"][topic] = {
                            "count": 0,
                            "avg_score": 0,
                            "total_score": 0
                        }
                    stats["topics"][topic]["count"] += 1
                    stats["topics"][topic]["total_score"] += combined_score
    
    # Calculate averages
    if stats["evaluated_messages"] > 0:
        stats["avg_scores"]["structure"] = structure_total / stats["evaluated_messages"]
        stats["avg_scores"]["content"] = content_total / stats["evaluated_messages"]
        stats["avg_scores"]["relevance"] = relevance_total / stats["evaluated_messages"]
        stats["avg_scores"]["combined"] = combined_total / stats["evaluated_messages"]
        
        # Calculate topic averages
        for topic in stats["topics"]:
            if stats["topics"][topic]["count"] > 0:
                stats["topics"][topic]["avg_score"] = stats["topics"][topic]["total_score"] / stats["topics"][topic]["count"]
    
    return jsonify(stats)