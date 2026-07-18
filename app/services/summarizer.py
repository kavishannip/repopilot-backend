import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.models.brief import RepoBrief
from app.services.llm import get_llm
from app.services.pipeline import ProcessingPipeline

class RepoSummarizer:
    def __init__(self):
        self.llm = get_llm().with_structured_output(RepoBrief)
        self.pipeline = ProcessingPipeline()

    async def generate_brief(self, repo_url: str) -> RepoBrief:
        """
        Generates a comprehensive repo brief using a layered approach.
        """
        # 1. Fetch metadata and ensure repo is processed/embedded
        print(f"Ensuring repo {repo_url} is processed...")
        process_stats = await self.pipeline.process_repo(repo_url)
        commit_sha = process_stats["commit_sha"]
        
        # 2. Get high level files (README, configs) via GithubClient directly or via search
        # To keep it robust, let's search for "README", "architecture", "setup", "configuration", "main entry point"
        print("Retrieving context chunks via similarity search...")
        
        queries = [
            "README or main documentation explaining the project purpose",
            "Configuration files, package.json, requirements.txt, Dockerfile",
            "Main entry point, setup, bootstrap, routing, or core architecture"
        ]
        
        context_chunks = []
        seen_ids = set()
        
        for query in queries:
            results = await self.pipeline.search_chunks(query, repo_url, top_k=3)
            for res in results:
                if res["id"] not in seen_ids:
                    seen_ids.add(res["id"])
                    context_chunks.append(res)
                    
        # 3. Format context
        context_text = ""
        for chunk in context_chunks:
            path = chunk['metadata'].get('path', 'Unknown file')
            chunk_type = chunk['metadata'].get('type', 'snippet')
            context_text += f"\n--- File: {path} ({chunk_type}) ---\n"
            context_text += chunk['text'][:1500] # cap chunk length just in case
            context_text += "\n"
            
        # 4. Generate summary
        print("Generating summary using Groq LLM...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert software architect analyzing a code repository.
Your task is to generate a comprehensive Repository Brief based on the provided code snippets and metadata.
You must return structured JSON matching the provided schema.

Make sure to include specific file paths in your entry_points and when discussing architecture.
Base your conclusions ONLY on the provided context. If something is missing, note it in flagged_issues."""),
            ("user", """Repository URL: {repo_url}
Commit: {commit_sha}

Here are the most relevant code chunks and documentation extracted from the repository:
{context}

Please analyze this and provide the structured RepoBrief.""")
        ])
        
        chain = prompt | self.llm
        brief = await chain.ainvoke({
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "context": context_text
        })
        
        return brief
