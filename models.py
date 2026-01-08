"""
Data models and type definitions for SmartPharma RAG.
Contains dataclasses for vector store and document representations.
"""

from dataclasses import dataclass
from typing import Any, List, Dict
from sentence_transformers import SentenceTransformer


@dataclass
class VectorStore:
    """
    Encapsulates FAISS index, text corpus, metadata, and embedder.
    Provides search functionality over the indexed documents.
    """
    index: Any  # FAISS index object
    texts: List[str]  # Original text documents
    metas: List[Dict[str, Any]]  # Metadata for each document
    embedder: SentenceTransformer  # Embedding model
    dim: int  # Dimension of embeddings

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Search for top-k most similar documents to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing rank, distance, text, and metadata
        """
        qv = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(qv, k)
        
        results = []
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append({
                "rank": rank + 1,
                "distance": float(D[0][rank]),
                "text": self.texts[idx],
                "meta": self.metas[idx],
            })
        return results