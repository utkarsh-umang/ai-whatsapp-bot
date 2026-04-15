"""
One-off: create a Gmail auth config in Composio and print the auth_config_id.

Run from project root:
  python3 scripts/create_gmail_auth_config.py

Requires COMPOSIO_API_KEY in .env (or export it in the shell).
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENV_PATH = os.path.join(_ROOT, ".env")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_PATH)
    except ImportError:
        if not os.path.isfile(_ENV_PATH):
            return
        with open(_ENV_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

from composio import Composio  # noqa: E402 — after env load


def main() -> None:
    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key:
        print("Set COMPOSIO_API_KEY in .env or export it.", file=sys.stderr)
        sys.exit(1)

    composio = Composio(api_key=api_key)
    auth_config = composio.auth_configs.create(
        toolkit="gmail",
        options={"type": "use_composio_managed_auth"},
    )
    print(auth_config.id)


if __name__ == "__main__":
    main()
