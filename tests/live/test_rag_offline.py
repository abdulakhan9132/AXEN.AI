import os
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge.embeddings import LocalEmbeddingProvider
from knowledge.ingest import ingest
from knowledge.retrieve import search_knowledge
from app.ai.model_manager import ModelManager

def main():
    print("--- 1. Testing Offline RAG Indexing ---")
    
    # Create test directory
    test_dir = Path("test_offline_rag_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test document
    doc_path = test_dir / "AXEN_TEST_KNOWLEDGE.txt"
    doc_path.write_text("AXEN is a local AI assistant whose language brain is trained from random initialization.")
    
    # Index it
    import knowledge.ingest as ingest_mod
    ingest_mod.DB_DIR = test_dir / "chroma_db"
    ingest([str(test_dir)])
    
    print("\n--- 2. Testing Offline RAG Retrieval ---")
    query = "What is AXEN?"
    print(f"Querying for: {query}")
    
    import knowledge.retrieve as retrieve_mod
    retrieve_mod.DB_DIR = test_dir / "chroma_db"
    results = search_knowledge(query, top_k=1)
    
    retrieved_text = results[0]["content"] if results else ""
    print(f"Retrieved Evidence: {retrieved_text}")
    assert "trained from random initialization" in retrieved_text, "Failed to retrieve correct evidence!"
    
    print("\n--- 3. Testing Own Brain Integration ---")
    # Start the Node.js server in the background
    server_process = subprocess.Popen(["node", "server.js"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2) # Wait for server to start
    
    try:
        # Use ModelManager which defaults to AxenOwnBrainProvider
        manager = ModelManager()
        model = manager.get_model()
        
        print(f"Using Model Provider: {model}")
        assert model.health_check(), "Own Brain is not healthy (Node.js server not responding)."
        
        # Pass evidence to the brain
        prompt = f"Evidence: {retrieved_text}\nQuestion: {query}"
        print(f"Sending to Brain: {prompt}")
        
        answer = model.generate(prompt)
        print(f"\nOwn Brain Answer:\n{answer}")
        
        print("\nSUCCESS! AXEN works fully offline using its own brain and local embeddings.")
    finally:
        server_process.terminate()
        
        # Cleanup
        # Note: on Windows, chromadb holds locks, so we may not be able to delete the directory immediately.
        try:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
