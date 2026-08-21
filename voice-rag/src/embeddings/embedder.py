import time
import numpy as np
import hashlib
import re
from typing import List, Union

class MultilingualEmbedder:
    """Multilingual dense embedding generator for Indic and English texts with fast feature vectorization."""
    def __init__(self, model_name: str = "BAAI/bge-m3", dimension: int = 1024):
        self.model_name = model_name
        self.dimension = dimension
        self.cache: dict[str, np.ndarray] = {}

    def _text_to_feature_vector(self, text: str) -> np.ndarray:
        """Deterministic feature vector using word and char n-grams hashing."""
        if not text:
            return np.zeros(self.dimension, dtype=np.float32)

        cache_key = text.strip()
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        words = re.findall(r'\w+', text.lower())
        vec = np.zeros(self.dimension, dtype=np.float32)
        
        if not words:
            return vec

        for word in words:
            # Word level hashing
            h_val = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = h_val % self.dimension
            sign = 1.0 if (h_val % 2 == 0) else -1.0
            vec[idx] += sign * 2.0
            
            # Sub-word char 3-grams for cross-lingual / morphological matching
            for i in range(len(word) - 2):
                gram = word[i:i+3]
                gh_val = int(hashlib.md5(gram.encode('utf-8')).hexdigest(), 16)
                g_idx = gh_val % self.dimension
                g_sign = 1.0 if (gh_val % 2 == 0) else -1.0
                vec[g_idx] += g_sign * 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if len(self.cache) < 2000:
            self.cache[cache_key] = vec

        return vec

    def encode(self, texts: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        """Encode text or list of texts into normalized embedding vectors."""
        if isinstance(texts, str):
            return self._text_to_feature_vector(texts)
            
        vecs = [self._text_to_feature_vector(t) for t in texts]
        return np.array(vecs, dtype=np.float32)

    def compute_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity scores between query and document vectors."""
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        if doc_embs.ndim == 1:
            doc_embs = doc_embs.reshape(1, -1)
        scores = np.dot(doc_embs, query_emb.T).squeeze(-1)
        return scores
