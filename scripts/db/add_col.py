import asyncio
import os
import sys

# Ensure the root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.database import engine
from sqlalchemy import text

async def add_col():
    async with engine.begin() as conn:
        await conn.execute(text('ALTER TABLE repositories ADD COLUMN IF NOT EXISTS brief JSONB;'))
        await conn.execute(text('ALTER TABLE repositories ADD COLUMN IF NOT EXISTS quick_summary TEXT;'))
    print('Done')

if __name__ == "__main__":
    asyncio.run(add_col())
