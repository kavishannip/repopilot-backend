import asyncio
from app.database import engine
from sqlalchemy import text

async def add_col():
    async with engine.begin() as conn:
        await conn.execute(text('ALTER TABLE repositories ADD COLUMN IF NOT EXISTS brief JSONB;'))
    print('Done')

if __name__ == "__main__":
    asyncio.run(add_col())
