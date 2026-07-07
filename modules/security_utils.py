import re

def sanitize_text(text: str, max_length: int = 300) -> str:
    """
    Sanitizes user input to prevent prompt injection and handle potential security risks.
    - Limits character length to prevent buffer/resource exhaustion.
    - Filters out typical prompt-injection keywords or override attempts.
    - Removes characters that could be used for injection.
    """
    if not isinstance(text, str):
        return ""
        
    # Enforce maximum length
    text = text[:max_length].strip()
    
    # Simple regex to check for prompt-injection markers or override attempts
    injection_patterns = [
        r"(ignore\s+(all\s+|the\s+|any\s+)?(previous|above|system)\s+instructions)",
        r"(system\s+prompt)",
        r"(you\s+are\s+now\s+a\s+)",
        r"(acting\s+as\s+a\s+)",
        r"(new\s+role)",
        r"(disregard\s+all\s+)",
        r"(override\s+instructions)"
    ]
    
    # Replace injection attempts with neutral text or flag them
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # Rather than crashing, we strip the injection trigger words
            text = re.sub(pattern, "[injection attempt blocked]", text, flags=re.IGNORECASE)
            
    # Remove potentially dangerous characters for standard output strings (like carriage returns, null bytes)
    text = text.replace("\0", "").replace("\r", "")
    
    return text

def validate_node_id(node_id: str, valid_ids: list) -> str:
    """
    Validates that a node ID is valid and exists in the stadium graph.
    Prevents injection or unexpected node routing.
    """
    if not isinstance(node_id, str):
        raise ValueError("Invalid node ID format")
    node_id = node_id.strip()
    if node_id not in valid_ids:
        raise ValueError(f"Node ID '{node_id}' does not exist in the stadium graph.")
    return node_id
