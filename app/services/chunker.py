import os
from typing import List, Dict, Any
from tree_sitter_languages import get_parser, get_language
from langchain_text_splitters import RecursiveCharacterTextSplitter

class CodeChunker:
    def __init__(self):
        # Fallback text chunker
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Supported AST languages
        self.supported_ast_languages = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx"
        }
        
        # Define tree-sitter queries for supported languages
        self.queries = {
            "python": """
                (function_definition) @function
                (class_definition) @class
            """,
            "javascript": """
                (function_declaration) @function
                (class_declaration) @class
                (arrow_function) @function
            """,
            "typescript": """
                (function_declaration) @function
                (class_declaration) @class
                (arrow_function) @function
            """,
            "tsx": """
                (function_declaration) @function
                (class_declaration) @class
                (arrow_function) @function
            """
        }
        
        self.parsers = {}
        for ext, lang_name in self.supported_ast_languages.items():
            if lang_name not in self.parsers:
                try:
                    self.parsers[lang_name] = get_parser(lang_name)
                except Exception as e:
                    print(f"Failed to load tree-sitter parser for {lang_name}: {e}")

    def chunk_file(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Chunks a file based on AST if supported, else falls back to text chunking.
        Returns a list of chunk dicts: {"text": str, "metadata": dict}
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # Check if we should use AST chunking
        lang_name = self.supported_ast_languages.get(ext)
        if lang_name and lang_name in self.parsers:
            try:
                return self._chunk_ast(file_path, content, lang_name)
            except Exception as e:
                # If AST parsing fails, fallback
                print(f"AST chunking failed for {file_path}, falling back to text. Error: {e}")
                
        return self._chunk_fallback(file_path, content)

    def _chunk_ast(self, file_path: str, content: str, lang_name: str) -> List[Dict[str, Any]]:
        parser = self.parsers[lang_name]
        language = get_language(lang_name)
        
        tree = parser.parse(content.encode('utf-8'))
        query_string = self.queries.get(lang_name)
        
        if not query_string:
            return self._chunk_fallback(file_path, content)
            
        query = language.query(query_string)
        captures = query.captures(tree.root_node)
        
        chunks = []
        last_end_byte = 0
        content_bytes = content.encode('utf-8')
        
        # Avoid overlapping chunks by skipping nodes inside already captured nodes
        captured_nodes = []
        for node, name in captures:
            is_inside = any(
                c_node.start_byte <= node.start_byte and c_node.end_byte >= node.end_byte
                for c_node, _ in captured_nodes
            )
            if not is_inside:
                captured_nodes.append((node, name))
                
        for node, name in captured_nodes:
            if node.start_byte > last_end_byte:
                loose_text = content_bytes[last_end_byte:node.start_byte].decode('utf-8', errors='ignore').strip()
                if len(loose_text) > 20: 
                    chunks.append({
                        "text": loose_text,
                        "metadata": {
                            "path": file_path,
                            "type": "module_level",
                            "chunking_mode": "ast",
                        }
                    })
            
            node_text = content_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            chunks.append({
                "text": node_text,
                "metadata": {
                    "path": file_path,
                    "type": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "chunking_mode": "ast",
                }
            })
            last_end_byte = node.end_byte
            
        if last_end_byte < len(content_bytes):
            loose_text = content_bytes[last_end_byte:].decode('utf-8', errors='ignore').strip()
            if len(loose_text) > 20:
                chunks.append({
                    "text": loose_text,
                    "metadata": {
                        "path": file_path,
                        "type": "module_level",
                        "chunking_mode": "ast",
                    }
                })
                
        if not chunks:
            return self._chunk_fallback(file_path, content)
            
        return chunks

    def _chunk_fallback(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        text_chunks = self.fallback_splitter.split_text(content)
        return [
            {
                "text": text,
                "metadata": {
                    "path": file_path,
                    "type": "text",
                    "chunking_mode": "fallback",
                }
            }
            for text in text_chunks
        ]
