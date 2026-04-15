import asyncio
from composio import Composio
import os
from dotenv import load_dotenv
import json

load_dotenv()
async def test():
    c = Composio()
    # just trying to see if tool exists or what error it gives
    try:
        res = c.tools.execute("GMAIL_FETCH_EMAILS", {}, dangerously_skip_version_check=True)
        print("Success")
    except Exception as e:
        print(e)
    
    # Try GMAIL_FIND_EMAIL
    try:
        res = c.tools.execute("GMAIL_SEARCH", {"q": "is:sent"}, dangerously_skip_version_check=True)
        print("Success Search")
    except Exception as e:
        print(e)


asyncio.run(test())
