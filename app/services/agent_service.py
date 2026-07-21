import os
import re
from typing import Dict, Any
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langgraph.prebuilt import create_react_agent
from github import Github, Auth

class GitHubAgentService:
    def __init__(self):
        # We initialize the LLM once
        model_name = os.getenv("REPO_ANALIZER_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        self.llm = ChatNVIDIA(model=model_name, temperature=0.2)
        
        self.system_prompt = """You are a highly capable software architecture analyst with access to GitHub tools.
Your task is to analyze the provided repository, search the codebase, read necessary files, and generate a highly accurate and comprehensive brief in Markdown format.
Focus on identifying:
1. Overall Architecture and patterns
2. Main entry points and key files
3. Data flow and component interactions
4. Tech stack and frameworks used
5. Any potential issues or technical debt

Only reference files and content you actually retrieved from the tools. Do not guess file paths.
Output the final analysis in well-structured Markdown."""

    def _extract_owner_repo(self, repo_url: str) -> str:
        """Extracts 'owner/repo' from a GitHub URL."""
        repo_url = repo_url.rstrip("/")
        pattern = r"github\.com/([^/]+/[^/]+)"
        match = re.search(pattern, repo_url)
        if match:
            return match.group(1).replace(".git", "")
        raise ValueError("Invalid GitHub URL")

    async def get_github_metadata(self, owner_repo: str) -> Dict[str, Any]:
        """Fetch raw metadata from GitHub using PyGithub."""
        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not token:
            raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN not set")
            
        auth = Auth.Token(token)
        g = Github(auth=auth)
        repo = g.get_repo(owner_repo)
        
        return {
            "full_name": repo.full_name,
            "description": repo.description,
            "homepage": repo.homepage,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "watchers": repo.watchers_count,
            "open_issues": repo.open_issues_count,
            "topics": repo.get_topics(),
            "license": repo.license.name if repo.license else "None",
            "created_at": repo.created_at.isoformat() if repo.created_at else None,
            "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
            "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
            "visibility": repo.visibility,
            "is_fork": repo.fork,
            "archived": repo.archived,
            "size_kb": repo.size,
            "primary_language": repo.language,
        }

    async def analyze_repo(self, repo_url: str) -> Dict[str, Any]:
        """Runs the LangChain agent and fetches metadata."""
        owner_repo = self._extract_owner_repo(repo_url)
        
        # Fetch metadata
        metadata = await self.get_github_metadata(owner_repo)
        
        # Initialize GitHub toolkit for this specific repo
        github_wrapper = GitHubAPIWrapper(
            github_repository=owner_repo
        )
        toolkit = GitHubToolkit.from_github_api_wrapper(github_wrapper)
        
        # Filter and rename tools for compatibility
        allowed_tools = {"Search code", "Read File"}
        tools = [t for t in toolkit.get_tools() if t.name in allowed_tools]
        for t in tools:
            t.name = t.name.replace(" ", "_").lower()
            
        # Create agent using langgraph's prebuilt react agent
        agent_executor = create_react_agent(self.llm, tools=tools, prompt=self.system_prompt)
        
        # Run agent
        input_msg = f"Analyze the repository '{owner_repo}'. Provide a highly accurate and comprehensive brief in Markdown format."
        result = await agent_executor.ainvoke({"messages": [("user", input_msg)]})
        
        return {
            "github_metadata": metadata,
            "repo_brief_markdown": result["messages"][-1].content
        }
