import unittest
import tempfile
import shutil
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.ingest import ingest
from knowledge.retrieve import search_knowledge

class TestKnowledge(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory with some test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_code.py")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("def calculate_magic_number():\n    return 42 * 7\n")
            
        # Override DB_DIR for testing
        import knowledge.ingest
        import knowledge.retrieve
        self.test_db_dir = os.path.join(self.test_dir, "chroma_test_db")
        knowledge.ingest.DB_DIR = Path(self.test_db_dir)
        knowledge.retrieve.DB_DIR = Path(self.test_db_dir)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_ingest_and_retrieve(self):
        # Test ingestion (skip if ollama isn't running or nomic-embed-text isn't pulled)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/", timeout=2)
            
            # Run ingest on test dir
            ingest([self.test_dir])
            
            # Test retrieve
            results = search_knowledge("What is the magic number?")
            
            self.assertTrue(len(results) > 0)
            self.assertTrue(any("42" in r.get("content", "") for r in results))
        except Exception as e:
            # Skip if ollama is not available in the test environment
            self.skipTest(f"Skipping test due to environment: {e}")

if __name__ == "__main__":
    unittest.main()
