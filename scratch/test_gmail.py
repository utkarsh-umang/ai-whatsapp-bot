import asyncio
from composio import Composio
import os
from dotenv import load_dotenv

load_dotenv()
def test():
    c = Composio()
    actions = c.apps.get("gmail")
    print(actions)

test()
