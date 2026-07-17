import os
import sys

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.chunker import CodeChunker

def test_chunker():
    chunker = CodeChunker()
    
    # 1. Test AST chunking on a Python snippet
    print("\n--- Test AST Chunking (Python) ---")
    py_content = """
import os

def hello_world():
    print("Hello world")
    
class MyClass:
    def method(self):
        pass

def another_func():
    return True
"""
    chunks = chunker.chunk_file("test.py", py_content)
    print(f"Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"\nChunk {i+1} Type: {c['metadata']['type']} | Mode: {c['metadata']['chunking_mode']}")
        print(repr(c['text']))
        
    # 2. Test fallback text chunking on unsupported language (e.g. Ruby)
    print("\n--- Test Fallback Text Chunking (Ruby) ---")
    rb_content = """
def hello_world
  puts "Hello world"
end
"""
    # Just repeating it to make it slightly longer
    rb_content = rb_content * 50
    chunks = chunker.chunk_file("test.rb", rb_content)
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunk 1 Mode: {chunks[0]['metadata']['chunking_mode']}")
    print(f"Chunk 1 Length: {len(chunks[0]['text'])}")

if __name__ == "__main__":
    test_chunker()
