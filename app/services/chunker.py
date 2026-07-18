import os
import re
from typing import List, Dict, Any, Optional
from tree_sitter_languages import get_parser, get_language
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# LangChain language-aware splitter map — broader coverage than tree-sitter alone
LANGCHAIN_LANGUAGE_MAP: Dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".cs": Language.CSHARP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".sol": Language.SOL,
    ".md": Language.MARKDOWN,
    ".html": Language.HTML,
    ".latex": Language.LATEX,
    ".proto": Language.PROTO,
}

# Import extraction patterns per language
IMPORT_PATTERNS: Dict[str, List[str]] = {
    "python": [
        r"^(?:from\s+[\w.]+\s+)?import\s+.+$",
    ],
    "javascript": [
        r"^import\s+.+from\s+['\"].+['\"]",
        r"^(?:const|let|var)\s+\w+\s*=\s*require\(['\"].+['\"]\)",
        r"^export\s+.+from\s+['\"].+['\"]",
    ],
    "typescript": [
        r"^import\s+.+from\s+['\"].+['\"]",
        r"^export\s+.+from\s+['\"].+['\"]",
    ],
    "go": [
        r'^import\s+\([\s\S]*?\)',
        r'^import\s+"[^"]+"',
    ],
}


class CodeChunker:
    def __init__(self):
        # Supported AST languages via tree-sitter
        self.supported_ast_languages = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
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
                (method_definition) @function
            """,
            "typescript": """
                (function_declaration) @function
                (class_declaration) @class
                (arrow_function) @function
                (method_definition) @function
                (interface_declaration) @class
                (type_alias_declaration) @class
            """,
            "tsx": """
                (function_declaration) @function
                (class_declaration) @class
                (arrow_function) @function
                (method_definition) @function
            """,
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
        Chunks a file using the best available strategy:
        1. AST-based chunking (tree-sitter) for Python/JS/TS
        2. LangChain language-aware splitting for other known languages
        3. Plain text fallback

        Returns list of {text, metadata} dicts.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Strategy 1: AST chunking (most precise)
        lang_name = self.supported_ast_languages.get(ext)
        if lang_name and lang_name in self.parsers:
            try:
                return self._chunk_ast(file_path, content, lang_name)
            except Exception as e:
                print(f"AST chunking failed for {file_path}, falling back. Error: {e}")

        # Strategy 2: LangChain language-aware splitting
        lc_lang = LANGCHAIN_LANGUAGE_MAP.get(ext)
        if lc_lang:
            try:
                return self._chunk_langchain_language(file_path, content, lc_lang)
            except Exception as e:
                print(f"LangChain language split failed for {file_path}: {e}")

        # Strategy 3: Plain text fallback
        return self._chunk_fallback(file_path, content)

    def extract_imports(self, file_path: str, content: str) -> List[str]:
        """
        Extract import/require statements from a file.
        These are stored in chunk metadata for dependency graph construction.
        """
        ext = os.path.splitext(file_path)[1].lower()
        lang_name = self.supported_ast_languages.get(ext, "")
        if ext in {".js", ".jsx"}:
            lang_name = "javascript"
        elif ext in {".ts", ".tsx"}:
            lang_name = "typescript"

        patterns = IMPORT_PATTERNS.get(lang_name, [])
        imports = []
        for line in content.splitlines():
            for pattern in patterns:
                if re.match(pattern, line.strip(), re.MULTILINE):
                    imports.append(line.strip())
                    break

        # Also extract from Python multi-line imports
        if lang_name == "python":
            # Handle "from x import (y, z)"
            multiline = re.findall(
                r"(?:from\s+[\w.]+\s+)?import\s+\([\s\S]*?\)",
                content, re.MULTILINE
            )
            imports.extend(m.strip() for m in multiline)

        return imports[:50]  # Cap to avoid polluting metadata

    def _chunk_ast(self, file_path: str, content: str, lang_name: str) -> List[Dict[str, Any]]:
        parser = self.parsers[lang_name]
        language = get_language(lang_name)

        tree = parser.parse(content.encode("utf-8"))
        query_string = self.queries.get(lang_name)

        if not query_string:
            return self._chunk_langchain_language(
                file_path, content,
                LANGCHAIN_LANGUAGE_MAP.get("." + lang_name[:2], None)
            ) or self._chunk_fallback(file_path, content)

        query = language.query(query_string)
        captures = query.captures(tree.root_node)

        chunks = []
        last_end_byte = 0
        content_bytes = content.encode("utf-8")

        # Extract imports for metadata
        imports = self.extract_imports(file_path, content)

        # Avoid overlapping chunks — skip nodes nested inside already-captured nodes
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
                loose_text = content_bytes[last_end_byte:node.start_byte].decode("utf-8", errors="ignore").strip()
                if len(loose_text) > 20:
                    chunks.append({
                        "text": loose_text,
                        "metadata": {
                            "path": file_path,
                            "type": "module_level",
                            "chunking_mode": "ast",
                            "imports": imports,
                        },
                    })

            node_text = content_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
            # Extract the name/identifier of the function/class if possible
            node_name = self._extract_node_name(node)
            chunks.append({
                "text": node_text,
                "metadata": {
                    "path": file_path,
                    "type": name,
                    "node_name": node_name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "chunking_mode": "ast",
                    "imports": imports if not chunks else [],  # only first chunk carries imports
                },
            })
            last_end_byte = node.end_byte

        if last_end_byte < len(content_bytes):
            loose_text = content_bytes[last_end_byte:].decode("utf-8", errors="ignore").strip()
            if len(loose_text) > 20:
                chunks.append({
                    "text": loose_text,
                    "metadata": {
                        "path": file_path,
                        "type": "module_level",
                        "chunking_mode": "ast",
                    },
                })

        if not chunks:
            return self._chunk_langchain_language(
                file_path, content,
                LANGCHAIN_LANGUAGE_MAP.get(os.path.splitext(file_path)[1].lower())
            ) or self._chunk_fallback(file_path, content)

        return chunks

    def _chunk_langchain_language(
        self,
        file_path: str,
        content: str,
        language: Optional[Language],
    ) -> List[Dict[str, Any]]:
        """Use LangChain's language-aware splitter with syntax-sensitive separators."""
        if language is None:
            return self._chunk_fallback(file_path, content)

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1200,
            chunk_overlap=150,
        )
        texts = splitter.split_text(content)
        imports = self.extract_imports(file_path, content)

        return [
            {
                "text": text,
                "metadata": {
                    "path": file_path,
                    "type": "code_block",
                    "chunking_mode": "langchain_language",
                    "imports": imports if i == 0 else [],
                },
            }
            for i, text in enumerate(texts)
        ]

    def _chunk_fallback(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Plain text fallback chunker."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        text_chunks = splitter.split_text(content)
        return [
            {
                "text": text,
                "metadata": {
                    "path": file_path,
                    "type": "text",
                    "chunking_mode": "fallback",
                },
            }
            for text in text_chunks
        ]

    def _extract_node_name(self, node) -> str:
        """Try to extract the identifier/name from an AST node (function/class name)."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="ignore")
        return ""
