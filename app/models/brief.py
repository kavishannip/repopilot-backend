from pydantic import BaseModel, Field
from typing import List

class FileReference(BaseModel):
    file_path: str = Field(description="The path to the file being referenced")
    description: str = Field(description="Why this file is important or what it contains")

class RepoBrief(BaseModel):
    tech_stack: List[str] = Field(description="List of main technologies, languages, and frameworks used in the repository")
    architecture_summary: str = Field(description="A concise summary of the overall architecture and how the components fit together")
    entry_points: List[FileReference] = Field(description="Key files that serve as entry points (e.g. main.py, index.js, app.py) or core configurations")
    flagged_issues: List[str] = Field(description="Any potential issues, technical debt, or areas lacking documentation identified during analysis")
    summary: str = Field(description="A high-level 1-2 paragraph summary of the repository's purpose and functionality")
