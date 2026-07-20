import re
from typing import List, Dict, Any, Tuple
from app.models.schema import GraphNode, GraphEdge
import uuid

# Very basic regexes for import extraction. Not meant to be a full AST parser,
# but enough for a fast lightweight scan.
PYTHON_IMPORT_RE = re.compile(r'^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))', re.MULTILINE)
TS_JS_IMPORT_RE = re.compile(r'(?:import\s+(?:.*?\s+from\s+)?|import\(\s*|require\(\s*)[\'"]([^\'"]+)[\'"]', re.MULTILINE)
JAVA_CS_SWIFT_RE = re.compile(r'^\s*(?:import|using)\s+([a-zA-Z0-9_\.]+)', re.MULTILINE)
RUST_RE = re.compile(r'^\s*(?:use|mod)\s+([a-zA-Z0-9_:]+)', re.MULTILINE)
C_CPP_RE = re.compile(r'^\s*#include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
PHP_RUBY_RE = re.compile(r'^\s*(?:require|require_relative|require_once|include|include_once|use)\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE)
GO_RE = re.compile(r'^\s*(?:import\s+)?(?:[a-zA-Z0-9_]+\s+)?[\'"]([^\'"]+)[\'"]', re.MULTILINE)

class StructuralScanner:
    """
    Lightweight scanner to build a structural graph of the repository.
    Creates folder/file nodes and contains/imports edges.
    """
    def __init__(self, max_files: int = 3000):
        self.max_files = max_files

    def _extract_imports(self, path: str, content: str) -> List[str]:
        """
        Extract imports from a file using fast regexes.
        """
        imports = []
        if path.endswith(".py"):
            for match in PYTHON_IMPORT_RE.finditer(content):
                module = match.group(1) or match.group(2)
                if module:
                    # Clean up things like 'app.services.xyz' -> 'app/services/xyz.py' approx,
                    # but since we only have string paths, we just keep the raw module string
                    # and try to match it later, or just store the raw import edge.
                    # For simplicity, we just store the raw module name for now, or attempt
                    # to map it to a path.
                    imports.append(module)
        elif path.endswith((".ts", ".tsx", ".js", ".jsx")):
            for match in TS_JS_IMPORT_RE.finditer(content):
                module = match.group(1)
                if module:
                    imports.append(module)
        elif path.endswith((".java", ".cs", ".swift", ".kt")):
            for match in JAVA_CS_SWIFT_RE.finditer(content):
                m = match.group(1)
                if m: imports.append(m)
        elif path.endswith(".rs"):
            for match in RUST_RE.finditer(content):
                m = match.group(1)
                if m: imports.append(m)
        elif path.endswith((".c", ".cpp", ".h", ".hpp", ".m")):
            for match in C_CPP_RE.finditer(content):
                m = match.group(1)
                if m: imports.append(m)
        elif path.endswith((".php", ".rb")):
            for match in PHP_RUBY_RE.finditer(content):
                m = match.group(1)
                if m: imports.append(m)
        elif path.endswith(".go"):
            if "import" in content:
                for match in GO_RE.finditer(content):
                    m = match.group(1)
                    if m: imports.append(m)
        return imports

    def scan(
        self,
        repository_id: uuid.UUID,
        all_files: List[Dict[str, Any]],
        file_contents: Dict[str, str],
        selected_paths: set,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Builds the graph nodes and edges.
        """
        nodes: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []

        # 1. Create nodes ONLY for actual files
        for f in all_files[:self.max_files]:
            path = f["path"]
            is_file = f.get("type", "file") in ("file", "blob") or "." in path.split("/")[-1]
            
            if not is_file:
                continue
                
            node_id = uuid.uuid4()
            nodes[path] = GraphNode(
                id=node_id,
                repository_id=repository_id,
                path=path,
                type="file",
                deep_analysis=(path in selected_paths)
            )
        
        # 'contains' edges removed to save space and focus entirely on code dependencies.

        # 2. Parse imports and create 'imports' edges
        for path, content in file_contents.items():
            if path not in nodes:
                continue
                
            source_node = nodes[path]
            raw_imports = self._extract_imports(path, content)
            
            # Try to resolve raw_imports to actual file nodes in the repo
            for raw_import in raw_imports:
                # Basic heuristic resolution:
                target_path = None
                
                if path.endswith(".py"):
                    # app.services.xyz -> app/services/xyz.py or app/services/xyz
                    converted = raw_import.replace(".", "/")
                    if f"{converted}.py" in nodes:
                        target_path = f"{converted}.py"
                    elif f"{converted}/__init__.py" in nodes:
                        target_path = f"{converted}/__init__.py"
                    elif converted in nodes:
                        target_path = converted
                
                elif path.endswith((".ts", ".tsx", ".js", ".jsx")):
                    # relative imports: ./xyz or ../xyz
                    if raw_import.startswith("."):
                        # resolve relative path
                        base_dir = path.rsplit("/", 1)[0] if "/" in path else ""
                        parts = base_dir.split("/") if base_dir else []
                        import_parts = raw_import.split("/")
                        for p in import_parts:
                            if p == ".":
                                continue
                            elif p == "..":
                                if parts: parts.pop()
                            else:
                                parts.append(p)
                        resolved = "/".join(parts)
                        
                        # check extensions
                        for ext in [".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js", ""]:
                            if f"{resolved}{ext}" in nodes:
                                target_path = f"{resolved}{ext}"
                                break
                    else:
                        # absolute/alias imports like 'src/components/xyz' or bare modules (ignore bare)
                        clean_import = raw_import
                        
                        # Strip common framework aliases (Vue, Svelte, Next, Nuxt)
                        for alias in ["@/", "~/", "$/", "@app/"]:
                            if clean_import.startswith(alias):
                                clean_import = clean_import[len(alias):]
                                break

                        # try direct match or prefixing with common root dirs
                        for ext in [".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js", ""]:
                            for base in ["", "src/", "app/", "lib/", "frontend/", "backend/"]:
                                test_path = f"{base}{clean_import}{ext}"
                                if test_path in nodes:
                                    target_path = test_path
                                    break
                            if target_path:
                                break
                else:
                    # Generic best-effort path mapping for other languages
                    clean_import = raw_import.replace(".", "/").replace("::", "/")
                    
                    test_exts = [""]
                    if path.endswith(".java"): test_exts = [".java"]
                    elif path.endswith(".cs"): test_exts = [".cs"]
                    elif path.endswith(".swift"): test_exts = [".swift"]
                    elif path.endswith(".kt"): test_exts = [".kt"]
                    elif path.endswith(".rs"): test_exts = [".rs"]
                    elif path.endswith((".c", ".cpp", ".h", ".hpp", ".m")): test_exts = ["", ".h", ".hpp", ".c", ".cpp", ".m"]
                    elif path.endswith(".php"): test_exts = [".php"]
                    elif path.endswith(".rb"): test_exts = [".rb"]
                    elif path.endswith(".go"): test_exts = [".go", "/"]

                    for ext in test_exts:
                        test_suffix = f"{clean_import}{ext}"
                        if test_suffix in nodes:
                            target_path = test_suffix
                            break
                        
                        # suffix check over all nodes (efficient enough for <3000 files)
                        for n_path in nodes:
                            if n_path.endswith(f"/{test_suffix}") or n_path == test_suffix:
                                target_path = n_path
                                break
                        if target_path:
                            break

                if target_path and target_path in nodes:
                    edges.append(
                        GraphEdge(
                            id=uuid.uuid4(),
                            repository_id=repository_id,
                            source_node_id=source_node.id,
                            target_node_id=nodes[target_path].id,
                            type="imports"
                        )
                    )

        return list(nodes.values()), edges
