import asyncio
from app.services.github_client import GitHubClient
from app.services.structural_scanner import StructuralScanner
import uuid

async def main():
    gh = GitHubClient()
    repo_url = "https://github.com/kavishannip/VisionTrace-PWA"
    meta = await gh.get_repo_metadata(repo_url)
    branch = meta["default_branch"]
    
    print("Fetching tree...")
    tree = await gh.get_file_tree_structure(repo_url, branch)
    all_files = tree["files"]
    
    scan_files = [
        f for f in all_files 
        if f.get("type", "file") in ("file", "blob") and f["path"].endswith((
            ".py", ".ts", ".tsx", ".js", ".jsx"
        ))
    ]
    
    print(f"Fetching {len(scan_files)} files content...")
    contents = await gh.get_files_content_batch(
        repo_url,
        [f["path"] for f in scan_files[:100]],
        branch=branch,
        max_chars_per_file=50000
    )
    
    scanner = StructuralScanner()
    nodes, edges = scanner.scan(uuid.uuid4(), all_files, contents, set())
    
    print(f"Generated {len(nodes)} nodes and {len(edges)} edges.")
    if len(edges) > 0:
        print(f"DEBUGGING EDGES:")
        for edge in edges[:5]:
            # find source and target nodes
            source = next((n for n in nodes if n.id == edge.source_node_id), None)
            target = next((n for n in nodes if n.id == edge.target_node_id), None)
            print(f"{source.path} imports {target.path}")
    else:
        print("NO EDGES FOUND!")

asyncio.run(main())
