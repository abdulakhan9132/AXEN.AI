import hashlib
import numpy as np

class LocalEmbeddingProvider:
    """
    A simple deterministic embedding provider that does not rely on any pretrained models.
    It uses a basic hashing trick to convert text into a fixed-size sparse-like dense vector,
    making it completely offline and independent of external AI models.
    """
    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def name(self) -> str:
        return "local_hash_embedding"

    def _hash_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            # Simple hash trick
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dimension
            vec[idx] += 1.0
        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def __call__(self, input: list[str]) -> list[list[float]]:
        # ChromaDB expects a __call__ method for embedding functions
        return [self._hash_text(text).tolist() for text in input]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # LangChain compatibility
        return self.__call__(texts)
    
    def embed_query(self, text: str) -> list[float]:
        # LangChain compatibility
        return self._hash_text(text).tolist()
