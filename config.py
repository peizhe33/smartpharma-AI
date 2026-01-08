"""
Configuration constants for SmartPharma RAG backend.
Contains all configurable paths, models, and system prompts.
"""

import os

# =========================
# DATA SOURCES
# =========================
DATA_DIR = r"C:\Users\peizh\OneDrive\Desktop\Datasets"
SOURCES = [
    os.path.join(DATA_DIR, "pubmedqa.jsonl"),
    os.path.join(DATA_DIR, "medqa.jsonl"),
    os.path.join(DATA_DIR, "nag_all.jsonl"),
]

# =========================
# INDEX CONFIGURATION
# =========================
INDEX_DIR = os.path.join(DATA_DIR, "index_med_rag")
INDEX_PATH = os.path.join(INDEX_DIR, "med_rag.index")
TEXTS_PATH = os.path.join(INDEX_DIR, "med_rag_texts.jsonl")

# =========================
# MODEL CONFIGURATION
# =========================
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# =========================
# OLLAMA CONFIGURATION
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2:0.5b"

# =========================
# SECTION MAPPING
# =========================
SECTION_MAP = {
    "A": {"NAG_A"},
    "B": {"NAG_B"},
    "ALL": {"NAG_A", "NAG_B"}
}

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PRIMER = (
    "You are SmartPharma Verification System.\n"
    "Goal: judge whether the diagnosis/indication aligns with Malaysian NAG evidence and standard antimicrobial practice.\n"
    "\n"
    "OUTPUT (max 6 lines, no markdown/bullets, no extra text):\n"
    "Verification: Diagnosis is accurate / Diagnosis is not fully accurate / Please review diagnosis to ensure its intended\n"
    "Confidence Score: <0-100%>  (MODEL confidence, not diagnosis accuracy)\n"
    "Explanation: 4-5 short sentences with [1][2] citations from retrieved context\n"
    "Citation: <[1] source, [2] source>\n"
    "\n"
    "DECISION RULE (internal, do not print): compute Diagnosis Accuracy Score 0-100%.\n"
    ">80 => Diagnosis is accurate | 50-80 => Diagnosis is not fully accurate | <50 => Please review diagnosis to ensure its intended.\n"
    "If key details are missing/unclear (age/weight/renal function/cultures/unclear dx), lower MODEL confidence.\n"
)

# =========================
# FLASK CONFIGURATION
# =========================
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5009
FLASK_DEBUG = False