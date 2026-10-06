from pathlib import Path
from typing import List, Dict

try:
    import chromadb
    from .embeddings import LocalEmbeddingProvider
    HAS_LANGCHAIN = True
except ImportError as e:
    HAS_LANGCHAIN = False
    print(f"Missing RAG dependency: {e}")

DB_DIR = Path(__file__).parent / "chroma_db"

def search_knowledge(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Retrieve top_k chunks from the local knowledge base matching the query."""
    if not HAS_LANGCHAIN:
        return [{"error": "Missing dependencies: pip install chromadb langchain langchain-community"}]
        
    if not DB_DIR.exists():
        return [{"error": "Knowledge database not found. Run ingest.py first."}]
        
    embeddings = LocalEmbeddingProvider()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        collection = client.get_collection(
            name="axen_knowledge",
            embedding_function=embeddings
        )
    except Exception:
        return [{"error": "Collection not found. Run ingest.py first."}]
        
    results = collection.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=top_k
    )
    
    docs = []
    if results and results.get('documents'):
        for i, text in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            docs.append({
                "source": meta.get('source', 'unknown'),
                "content": text
            })
    return docs

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the AIModel interface work?"
    results = search_knowledge(query)
    for i, res in enumerate(results):
        if "error" in res:
            print(res["error"])
        else:
            print(f"\n--- Match {i+1} ({res['source']}) ---")
            print(res['content'][:300] + "...\n")
