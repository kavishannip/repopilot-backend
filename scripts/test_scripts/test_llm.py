import os
import sys

# Ensure the root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel, Field

class TestModel(BaseModel):
    architecture_summary: str = Field(description="Architecture summary")
    tech_stack: list = Field(description="Tech stack")

import json
try:
    obj = TestModel.model_validate_json('{}')
    print("Parsed empty:", obj)
except Exception as e:
    print("Error parsing empty:", type(e).__name__)
