import os
from motor.motor_asyncio import AsyncIOMotorClient
from models import UserProfile, ChatMessage
from typing import Optional
from datetime import datetime

_client: AsyncIOMotorClient = None
MAX_HISTORY_TURNS = 20   # how many turns to load per conversation


def get_db():
    return _client["poke_v2"]


async def connect():
    global _client
    _client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27047"))


async def disconnect():
    if _client:
        _client.close()


# ── User helpers ──────────────────────────────────────────────────────

async def get_user_by_entity(entity_id: str) -> Optional[dict]:
    return await get_db().users.find_one({"entity_id": entity_id})


async def get_user_by_phone(phone: str) -> Optional[dict]:
    return await get_db().users.find_one({"phone": phone})


async def upsert_user(entity_id: str, update: dict):
    await get_db().users.update_one(
        {"entity_id": entity_id},
        {"$set": update},
        upsert=True,
    )


async def set_status(entity_id: str, status: str):
    await upsert_user(entity_id, {"status": status})


async def save_profile(entity_id: str, profile: UserProfile):
    await upsert_user(entity_id, {
        "profile": profile.model_dump(),
        "name": profile.name,
        "email": profile.email,
    })


# ── Conversation history ──────────────────────────────────────────────
#
# Each document in `conversations`:
#   { phone, role, content, ts }
#
# We store flat messages (not nested) so we can query latest N efficiently.

def _conversation_query(user: dict) -> dict:
    """WhatsApp users keyed by phone; web chat keyed by entity_id."""
    if user.get("phone"):
        return {"phone": user["phone"]}
    return {"entity_id": user["entity_id"]}


async def append_message(phone: str, role: str, content: str):
    """Legacy: phone-only conversations (WhatsApp)."""
    await get_db().conversations.insert_one({
        "phone": phone,
        "role": role,
        "content": content,
        "ts": datetime.utcnow(),
    })


async def append_message_for_user(user: dict, role: str, content: str):
    q = _conversation_query(user)
    doc = {**q, "role": role, "content": content, "ts": datetime.utcnow()}
    await get_db().conversations.insert_one(doc)


async def get_history(phone: str) -> list[dict]:
    """Returns last MAX_HISTORY_TURNS messages, oldest first (phone-keyed)."""
    cursor = (
        get_db().conversations
        .find({"phone": phone}, {"_id": 0, "role": 1, "content": 1})
        .sort("ts", -1)
        .limit(MAX_HISTORY_TURNS)
    )
    docs = await cursor.to_list(length=MAX_HISTORY_TURNS)
    return list(reversed(docs))


async def get_history_for_user(user: dict) -> list[dict]:
    """Last N messages for this user (WhatsApp or web)."""
    q = _conversation_query(user)
    cursor = (
        get_db().conversations
        .find(q, {"_id": 0, "role": 1, "content": 1})
        .sort("ts", -1)
        .limit(MAX_HISTORY_TURNS)
    )
    docs = await cursor.to_list(length=MAX_HISTORY_TURNS)
    return list(reversed(docs))
