import asyncio
from composio import Composio
import os
from dotenv import load_dotenv

load_dotenv()
async def test():
    c = Composio()
    res = c.tools.execute("COMPOSIO_SEARCH", {"query": "Vercel"}, dangerously_skip_version_check=True)
    print(res)

asyncio.run(test())
