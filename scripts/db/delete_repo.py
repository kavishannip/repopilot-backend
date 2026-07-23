import asyncio
import argparse
import os
import sys

# Ensure the root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy.future import select
from sqlalchemy import delete
from app.database import async_session
from app.models.schema import Repository, AnalysisJob, RepoChunk, GraphNode, GraphEdge

from sqlalchemy.exc import ProgrammingError

async def delete_all():
    async with async_session() as session:
        models = [
            ("AnalysisJob", AnalysisJob), 
            ("GraphEdge", GraphEdge), 
            ("GraphNode", GraphNode), 
            ("RepoChunk", RepoChunk), 
            ("Repository", Repository)
        ]
        
        for name, model in models:
            try:
                await session.execute(delete(model))
                await session.commit()
            except ProgrammingError as e:
                await session.rollback()
                if "does not exist" in str(e):
                    print(f"Table for {name} does not exist, skipping.")
                else:
                    raise
            except Exception as e:
                await session.rollback()
                raise
                
        print("Successfully deleted all repositories, quick summaries, chunks, nodes, edges, and analysis jobs from the database (if tables existed).")

async def delete_by_url(repo_url: str):
    async with async_session() as session:
        # Delete analysis jobs for this repo URL
        try:
            await session.execute(delete(AnalysisJob).where(AnalysisJob.repo_url == repo_url))
            await session.commit()
        except ProgrammingError as e:
            await session.rollback()
            if "does not exist" not in str(e):
                raise
            print("Table for AnalysisJob does not exist, skipping.")
        
        try:
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            result = await session.execute(stmt)
            repo = result.scalar_one_or_none()
            
            if repo:
                await session.delete(repo)
                await session.commit()
                print(f"Successfully deleted repository, quick summaries, chunks, nodes, edges, and its jobs: {repo_url}")
            else:
                await session.commit()  # commit job deletion even if repo entry didn't exist
                print(f"Repository not found: {repo_url} (deleted any matching analysis jobs)")
        except ProgrammingError as e:
            await session.rollback()
            if "does not exist" not in str(e):
                raise
            print("Table for Repository does not exist, skipping.")

def main():
    parser = argparse.ArgumentParser(description="Delete repository data from the database.")
    parser.add_argument("--all", action="store_true", help="Delete all repositories")
    parser.add_argument("--url", type=str, help="Delete a specific repository by URL")
    
    args = parser.parse_args()
    
    if args.all:
        asyncio.run(delete_all())
    elif args.url:
        asyncio.run(delete_by_url(args.url))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
