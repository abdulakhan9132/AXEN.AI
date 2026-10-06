import unittest
import os
from pathlib import Path

# Add project root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.prepare_dataset import detect_language, has_code, estimate_tokens

class TestPrepareDataset(unittest.TestCase):
    def test_detect_language_fence(self):
        text = "Here is some code:\n```python\nprint('hello')\n```"
        self.assertEqual(detect_language(text), "python")
        
        text2 = "```js\nconsole.log(1);\n```"
        self.assertEqual(detect_language(text2), "javascript")
        
    def test_detect_language_heuristics(self):
        text = "def my_func(self):\n    return True"
        self.assertEqual(detect_language(text), "python")
        
        text2 = "public static void main(String[] args) {}"
        self.assertEqual(detect_language(text2), "java")
        
        text3 = "const obj = () => { return 1; }"
        self.assertEqual(detect_language(text3), "javascript")

    def test_has_code(self):
        self.assertTrue(has_code("Here is code: ```python print(1) ```"))
        self.assertFalse(has_code("No code here."))

    def test_estimate_tokens(self):
        # 20 chars / 4 = 5
        self.assertEqual(estimate_tokens("12345678901234567890"), 5)

if __name__ == "__main__":
    unittest.main()
