from composio import Composio
import os
os.environ["COMPOSIO_API_KEY"] = "sk-placeholder"
c = Composio()
try:
    c.tools.execute(
        "GMAIL_GET_PROFILE",
        {},
        user_id="123",
        dangerously_skip_version_check=True,
    )
except Exception as e:
    import traceback
    traceback.print_exc()
