import os
import httpx

PERISKOPE_API_KEY = os.getenv("PERISKOPE_API_KEY")
# Your bot's WhatsApp number with country code, no special chars — e.g. 919876543210
# Used in the x-phone header to tell Periskope which connected phone to send from
PERISKOPE_PHONE = os.getenv("PERISKOPE_PHONE")
PERISKOPE_BASE = "https://api.periskope.app/v1"


def _chat_id(phone: str) -> str:
    """
    Periskope expects chat_id in the format 919876543210@c.us for 1-on-1 chats.
    Strip any existing @c.us suffix before re-adding so we're idempotent.
    """
    clean = phone.replace("@c.us", "").replace("@g.us", "").strip()
    return f"{clean}@c.us"


async def send(to_phone: str, text: str) -> bool:
    """
    Send a WhatsApp text message via Periskope.

    Auth requires TWO headers:
      Authorization: Bearer <api_key>
      x-phone: <your_bot_phone>   ← directs request to the right connected phone

    Body shape: { chat_id, message }
      chat_id for 1-on-1: "919876543210@c.us"  (the RECIPIENT's number)
      message: plain text (supports WhatsApp markdown: *bold*, _italic_)
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{PERISKOPE_BASE}/messages",
            headers={
                "Authorization": f"Bearer {PERISKOPE_API_KEY}",
                "x-phone": PERISKOPE_PHONE,
                "Content-Type": "application/json",
            },
            json={
                "chat_id": _chat_id(to_phone),
                "message": text,
            },
        )
        resp.raise_for_status()
        return True
