import asyncio
import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.config import settings
from app.services.github_client import GitHubClient

logger = logging.getLogger(__name__)

# Module-level state for tools (since @tool decorators don't easily bind instances)
# We will use a context var or simple class wrapper to bind state.
class GitHubToolkitAsync:
    def __init__(self, github: GitHubClient, repo_url: str, branch: str):
        self.github = github
        self.repo_url = repo_url
        self.branch = branch
        self._list_cache: Optional[List[Dict[str, Any]]] = None
        self._content_cache: Dict[str, str] = {}
        self.read_files: Dict[str, str] = {} # Files the agent specifically requested to read

    async def _get_all_files(self):
        if self._list_cache is None:
            self._list_cache = await self.github.list_files(self.repo_url, self.branch)
        return self._list_cache

    def get_tools(self):
        
        @tool
        async def list_files(directory: str = "") -> str:
            """
            List all files and directories in the repository, optionally filtered by a specific directory path.
            Use this to understand the structure of the repository.
            """
            all_files = await self._get_all_files()
            
            dir_prefix = directory
            if dir_prefix and not dir_prefix.endswith('/'):
                dir_prefix += '/'
                
            results = []
            for f in all_files:
                path = f["path"]
                if dir_prefix and not path.startswith(dir_prefix):
                    continue
                results.append(path)
                
            if not results:
                return f"No files found in directory: {directory}"
            
            # Truncate if too many files to avoid blowing up context
            if len(results) > 200:
                return f"Found {len(results)} files. Showing first 200:\n" + "\n".join(results[:200])
            return "\n".join(results)

        @tool
        async def get_file_contents(file_path: str) -> str:
            """
            Get the full contents of a specific file in the repository.
            You must provide the exact file path as returned by list_files.
            """
            if file_path in self._content_cache:
                return self._content_cache[file_path]
                
            # Fetch content
            contents = await self.github.get_files_content_batch(
                self.repo_url, [file_path], self.branch, max_chars_per_file=50_000
            )
            
            if file_path not in contents:
                return f"Error: File '{file_path}' not found or could not be read."
                
            content = contents[file_path]
            self._content_cache[file_path] = content
            self.read_files[file_path] = content # Record that the agent read it
            
            return f"--- {file_path} ---\n{content}"

        @tool
        async def get_multiple_files(file_paths: List[str]) -> str:
            """
            Get the full contents of multiple files in a single call.
            Pass a list of exact file paths.
            """
            to_fetch = [p for p in file_paths if p not in self._content_cache]
            if to_fetch:
                contents = await self.github.get_files_content_batch(
                    self.repo_url, to_fetch, self.branch, max_chars_per_file=50_000
                )
                for p, c in contents.items():
                    self._content_cache[p] = c
                    self.read_files[p] = c
                    
            results = []
            for path in file_paths:
                if path in self._content_cache:
                    results.append(f"--- {path} ---\n{self._content_cache[path]}")
                else:
                    results.append(f"Error: File '{path}' not found.")
                    
            return "\n\n".join(results)
            
        return [list_files, get_file_contents, get_multiple_files]


class AgentExplorer:
    def __init__(self, github: GitHubClient):
        self.github = github
        # We need a 70B model or equivalent for agent loops.
        # Assuming GROQ_MODEL might be 8B, we'll try to use llama3-70b-8192 if possible, or fallback to default
        model_name = "llama3-70b-8192" 
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=model_name,
            temperature=0.0,
            max_tokens=2048,
        )

    async def explore(
        self, 
        repo_url: str, 
        branch: str, 
        hints: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Runs the agentic exploration loop.
        Returns a dict mapping file paths to their full text content read by the agent.
        """
        toolkit = GitHubToolkitAsync(self.github, repo_url, branch)
        tools = toolkit.get_tools()
        
        system_prompt = f"""You are an expert software architect analyzing a repository.
Your goal is to explore the repository and read the most critical files to understand its architecture, data flow, and components.
You have been provided with some pre-computed hints to guide your exploration. Do NOT explore blindly.

HINTS:
Frameworks detected: {hints.get('frameworks', [])}
Critical files: {hints.get('classification', {}).get('critical', [])}
Entry points: {hints.get('classification', {}).get('entry_points', [])}

Repository tree summary:
{hints.get('ascii_tree', '')}

INSTRUCTIONS:
1. Use list_files to explore directories of interest.
2. Use get_file_contents or get_multiple_files to read the actual code.
3. Focus on reading entry points, API routes, database schemas, and core business logic.
4. Once you have read enough files to understand the core architecture (or if you run out of things to check), use the Final Answer to summarize briefly what you found. The actual files you read are automatically collected in the background.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        # Max steps = 20, max_execution_time = 120s (2 mins)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=20, 
            max_execution_time=120,
            return_intermediate_steps=False
        )

        try:
            logger.info(f"Starting agent exploration for {repo_url}...")
            # We run it in async mode
            await agent_executor.ainvoke({"input": "Begin exploration."})
        except Exception as e:
            logger.error(f"Agent exploration failed or timed out: {e}")
            
        logger.info(f"Agent finished. Read {len(toolkit.read_files)} files.")
        return toolkit.read_files
