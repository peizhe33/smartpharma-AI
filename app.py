"""
Flask REST API for SmartPharma RAG backend.
Main entry point that orchestrates retrieval and LLM generation.
"""

import time
from flask import Flask, request, jsonify, make_response

from config import (
    SECTION_MAP, TOP_K, OLLAMA_MODEL,
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG
)
from vector_store import build_or_load_store
from retrieval import filter_by_section, rank_with_source_bonus
from prompt_builder import make_prompt, ensure_verification_line
from llm_client import call_ollama

# Initialize CORS if available
try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False


# =========================
# Initialize Flask app
# =========================
app = Flask(__name__)

if _HAS_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_cors_headers(resp):
    """Ensure CORS headers are present on all responses."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# =========================
# Load vector store
# =========================
print("Initializing vector store...")
STORE = build_or_load_store()
print(f"Vector store ready with {len(STORE.texts)} documents")


# =========================
# Routes
# =========================
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "docs": len(STORE.texts),
        "model": OLLAMA_MODEL
    })


@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    """
    Main RAG endpoint.
    Accepts POST with JSON: {question, age?, section?, k?}
    Returns JSON: {question, section, answer, retrieved, elapsed_s}
    """
    # Handle preflight
    if request.method == "OPTIONS":
        resp = make_response("", 200)
        req_headers = request.headers.get("Access-Control-Request-Headers", "")
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = (
            req_headers if req_headers else "Content-Type, Authorization"
        )
        return resp

    t0 = time.time()

    # Parse request
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    k = int(data.get("k", TOP_K))

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Determine section from age or explicit section parameter
    age_raw = data.get("age", None)
    explicit_section = (data.get("section") or "").strip().upper()
    section = None

    # Priority 1: Age-based section selection
    if age_raw is not None and str(age_raw).strip() != "":
        try:
            age = int(age_raw)
            section = "A" if age >= 18 else "B"
        except ValueError:
            section = None

    # Priority 2: Explicit section parameter
    if section is None and explicit_section in SECTION_MAP:
        section = explicit_section

    # Priority 3: Default to ALL if unknown
    if section is None:
        section = "ALL"

    allowed = SECTION_MAP.get(section, SECTION_MAP["ALL"])

    # Build query (add age note for model if needed)
    retrieval_query = question
    prompt_question = question
    
    if section == "ALL":
        prompt_question += (
            "\n\nNOTE: Patient age not provided. "
            "Ask for age to choose NAG A (adult) vs NAG B (peds)."
        )

    try:
        # Retrieve documents
        retrieved_all = STORE.search(retrieval_query, max(k * 3, k + 5))
        retrieved = filter_by_section(retrieved_all, allowed)

        # Fallback to re-ranking if no section-filtered results
        if retrieved:
            retrieved = retrieved[:k]
        else:
            if section in ("A", "B"):
                retrieved = rank_with_source_bonus(
                    retrieved_all, 
                    SECTION_MAP[section]
                )[:k]
            else:
                retrieved = retrieved_all[:k]

        # Generate answer
        prompt = make_prompt(prompt_question, retrieved)
        answer = call_ollama(prompt)
        answer = ensure_verification_line(answer)

        elapsed = round(time.time() - t0, 3)

        return jsonify({
            "question": question,
            "section": section,
            "answer": answer,
            "retrieved": retrieved,
            "elapsed_s": elapsed
        })

    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        return jsonify({
            "error": f"Backend failed: {e}",
            "elapsed_s": elapsed
        }), 500


# =========================
# Main entry point
# =========================
if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)