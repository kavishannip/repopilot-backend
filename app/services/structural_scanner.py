import re
from typing import List, Dict, Any, Tuple
from app.models.schema import GraphNode, GraphEdge
import uuid

# Very basic regexes for import extraction. Not meant to be a full AST parser,
# but enough for a fast lightweight scan.
PYTHON_IMPORT_RE = re.compile(r'^(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))', re.MULTILINE)
TS_JS_IMPORT_RE = re.compile(r'import\s+(?:.*?\s+from\s+)?[\'"]([^\'"]+)[\'"]', re.MULTILINE)

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

        # 1. Create nodes for all folders and files
        for f in all_files[:self.max_files]:
            path = f["path"]
            is_file = f.get("type", "file") == "file" or "." in path.split("/")[-1]
            
            # Create node for the file
            node_id = uuid.uuid4()
            nodes[path] = GraphNode(
                id=node_id,
                repository_id=repository_id,
                path=path,
                type="file" if is_file else "folder",
                deep_analysis=(path in selected_paths)
            )

            # Create nodes for parent folders and 'contains' edges
            parts = path.split("/")
            for i in range(1, len(parts)):
                folder_path = "/".join(parts[:i])
                if folder_path not in nodes:
                    folder_id = uuid.uuid4()
                    nodes[folder_path] = GraphNode(
                        id=folder_id,
                        repository_id=repository_id,
                        path=folder_path,
                        type="folder",
                        deep_analysis=False
                    )
                
                # edge from parent folder to this item (folder or file)
                # the item path is "/".join(parts[:i+1])
                child_path = "/".join(parts[:i+1])
                # We will add edges in a second pass or check if we already added it
        
        # Build contains edges
        for path, node in nodes.items():
            if "/" in path:
                parent_path = path.rsplit("/", 1)[0]
                if parent_path in nodes:
                    edges.append(
                        GraphEdge(
                            id=uuid.uuid4(),
                            repository_id=repository_id,
                            source_node_id=nodes[parent_path].id,
                            target_node_id=node.id,
                            type="contains"
                        )
                    )

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
                        # just try direct match
                        for ext in [".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js", ""]:
                            if f"{raw_import}{ext}" in nodes:
                                target_path = f"{raw_import}{ext}"
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
