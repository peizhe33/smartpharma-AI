"""
Vector store management for SmartPharma RAG.
Handles building and loading FAISS indices with sentence embeddings.
"""

import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from config import INDEX_PATH, TEXTS_PATH, INDEX_DIR, EMBEDDER_MODEL, SOURCES
from models import VectorStore
from utils import load_jsonl, ensure_directory


def build_or_load_store() -> VectorStore:
    """
    Build a new FAISS index or load existing one from disk.
    
    Returns:
        VectorStore instance with loaded/built index
        
    Raises:
        RuntimeError: If no documents found or cached index is invalid
    """
    ensure_directory(INDEX_DIR)
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    # Try loading existing index
    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        print("Loading existing FAISS index + texts ...")
        index = faiss.read_index(INDEX_PATH)
        
        texts, metas = [], []
        with open(TEXTS_PATH, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
                metas.append(obj.get("meta") or {})
                
        if not texts:
            raise RuntimeError("Cached TEXTS_PATH is empty. Delete index folder and rebuild.")
            
        return VectorStore(
            index=index,
            texts=texts,
            metas=metas,
            embedder=embedder,
            dim=index.d
        )

    # Build new index
    print("Building index from sources ...")
    raw_docs = load_jsonl(SOURCES)
    
    if not raw_docs:
        raise RuntimeError("No documents found in SOURCES. Check your JSONL paths.")

    texts = [d["text"] for d in raw_docs]
    metas = [d.get("meta") or {} for d in raw_docs]

    # Generate embeddings and create index
    X = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    # Save to disk
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        for t, m in zip(texts, metas):
            f.write(json.dumps({"text": t, "meta": m}, ensure_ascii=False) + "\n")

    print(f"Built and saved index with {len(texts)} docs, dim={index.d}")
    
    return VectorStore(
        index=index,
        texts=texts,
        metas=metas,
        embedder=embedder,
        dim=index.d
    )