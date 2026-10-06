import os
from pathlib import Path

try:
    import chromadb
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from .embeddings import LocalEmbeddingProvider
    HAS_LANGCHAIN = True
except ImportError as e:
    HAS_LANGCHAIN = False
    print(f"Missing RAG dependency: {e}")

# Default to indexing the root project
PROJECT_ROOT = Path(__file__).parent.parent
DB_DIR = Path(__file__).parent / "chroma_db"

def ingest(source_dirs: list[str] = [str(PROJECT_ROOT)]):
    if not HAS_LANGCHAIN:
        print("Please install requirements: pip install chromadb langchain langchain-community")
        return
        
    print("Initializing embedding model (local hash-based)...")
    embeddings = LocalEmbeddingProvider()
    
    docs = []
    print(f"Loading documents from {source_dirs}...")
    for src in source_dirs:
        loaders = {
            ".py": DirectoryLoader(src, glob="**/*.py", loader_cls=TextLoader, exclude=["**/__pycache__/**", "**/venv/**"]),
            ".js": DirectoryLoader(src, glob="**/*.js", loader_cls=TextLoader, exclude=["**/node_modules/**"]),
            ".md": DirectoryLoader(src, glob="**/*.md", loader_cls=TextLoader),
            ".txt": DirectoryLoader(src, glob="**/*.txt", loader_cls=TextLoader)
        }
        
        for ext, loader in loaders.items():
            try:
                loaded = loader.load()
                docs.extend(loaded)
                print(f"Loaded {len(loaded)} {ext} files from {src}")
            except Exception as e:
                print(f"Warning: Could not load {ext} files from {src}: {e}")

    if not docs:
        print("No documents found.")
        return

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"Storing in Chroma DB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name="axen_knowledge",
        embedding_function=embeddings
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
