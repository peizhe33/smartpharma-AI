"""
Prompt building and text normalization for SmartPharma RAG.
Handles context assembly and response post-processing.
"""

import re
from typing import List, Dict, Any
from config import SYSTEM_PRIMER


def make_prompt(user_question: str, retrieved: List[Dict[str, Any]]) -> str:
    """
    Assemble RAG prompt with system instructions, context, and user question.
    
    Args:
        user_question: User's input question
        retrieved: List of retrieved document dictionaries
        
    Returns:
        Complete prompt string for the LLM
    """
    ctx_lines = []
    
    for item in retrieved:
        m = item.get("meta", {})
        src = m.get("source", "Source")
        ident = m.get("pmid") or m.get("id") or m.get("url") or ""
        title = m.get("title") or ""
        
        # Build header line
        header = f"{src}"
        if title:
            header += f" — {title}"
        if ident:
            header += f" ({ident})"
        
        # Truncate long snippets
        snippet = item["text"].strip().replace("\n\n", "\n")
        if len(snippet) > 700:
            snippet = snippet[:700] + " …"
        
        ctx_lines.append(f"[{item['rank']}] {header}\n{snippet}\n")

    context_block = "\n".join(ctx_lines) if ctx_lines else "No context retrieved."
    
    prompt = (
        f"{SYSTEM_PRIMER}\n\n"
        f"Retrieved context (use to ground your answer; cite [#] where used):\n"
        f"{context_block}\n\n"
        f"User question:\n{user_question}\n\n"
        f"Write ONLY the required fields exactly in the requested format."
    )
    
    return prompt


# Regex for finding verification lines
_RE_VERIF = re.compile(
    r"^\s*(?:\*{1,2}\s*)?verification(?:\s*\*{1,2})?\s*:\s*(.+)$",
    re.I | re.M,
)


def normalize_labels(text: str) -> str:
    """
    Remove markdown formatting from field labels.
    
    Args:
        text: Raw LLM output
        
    Returns:
        Text with normalized labels
    """
    return re.sub(
        r"^\s*\*{1,2}\s*(Verification|Guideline Used|Confidence Score|Explanation|Citation)\s*\*{1,2}\s*:",
        r"\1:",
        text,
        flags=re.I | re.M,
    )


def ensure_verification_line(text: str) -> str:
    """
    Ensure response has exactly one Verification line, add default if missing.
    Handles duplicate verification lines by keeping only the last one.
    
    Args:
        text: Raw LLM output
        
    Returns:
        Normalized text with single Verification line
    """
    text = normalize_labels(text)

    lines = text.splitlines()
    
    # Find all verification line indices
    verif_idxs = [
        i for i, line in enumerate(lines)
        if re.match(r"^\s*Verification\s*:", line, re.I)
    ]
    
    # If multiple verification lines, keep only the last one
    if len(verif_idxs) > 1:
        keep = verif_idxs[-1]
        lines = [
            line for i, line in enumerate(lines) 
            if i == keep or i not in verif_idxs
        ]
        text = "\n".join(lines)

    # If no verification line found, add default
    if not _RE_VERIF.search(text):
        text = "Verification: Please review diagnosis to ensure its intended\n" + text

    return text