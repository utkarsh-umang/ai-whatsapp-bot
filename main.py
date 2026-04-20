import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from composio import Composio

import db
from models import (
    ChatSendRequest,
    ChatSendResponse,
)
from enrichment import run_enrichment
from chat import craft_first_message, reply

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
# Create this in Composio dashboard → Auth Configs → Gmail → copy the config ID
COMPOSIO_GMAIL_AUTH_CONFIG_ID = os.getenv("COMPOSIO_GMAIL_AUTH_CONFIG_ID")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pages ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Composio OAuth ────────────────────────────────────────────────────

@app.get("/auth/composio/start")
async def composio_start(entity_id: str = None):
    if not entity_id:
        entity_id = str(uuid.uuid4())

    await db.upsert_user(entity_id, {"entity_id": entity_id, "status": "pending_auth"})

    composio = Composio(api_key=COMPOSIO_API_KEY)
    req = await asyncio.to_thread(
        composio.connected_accounts.link,
        entity_id,
        COMPOSIO_GMAIL_AUTH_CONFIG_ID,
        callback_url=f"{BASE_URL}/auth/composio/callback?entity_id={entity_id}",
    )
    return {"redirect_url": req.redirect_url, "entity_id": entity_id}


@app.get("/auth/composio/callback")
async def composio_callback(entity_id: str, background_tasks: BackgroundTasks):
    """Gmail OAuth is done. Fire enrichment in background; user goes straight to chat."""
    await db.set_status(entity_id, "enriching")

    # Start enrichment immediately — usually ready before they open chat
    background_tasks.add_task(_run_enrichment_task, entity_id)

    return RedirectResponse(f"/?step=2&entity_id={entity_id}")


import logging
logging.basicConfig(level=logging.INFO)

async def _run_enrichment_task(entity_id: str):
    """Runs enrichment, stores profile. Silently fails — user won't know."""
    logging.info(f"Started background enrichment for entity: {entity_id}")
    try:
        profile = await run_enrichment(entity_id)
        await db.save_profile(entity_id, profile)
        logging.info(f"Successfully completed enrichment for entity: {entity_id}. Profile: {profile.name}")
    except Exception as e:
        # Store a minimal profile so the bot still works
        logging.error(f"Enrichment failed for {entity_id}: {e}", exc_info=True)
    finally:
        # Unblock chat, but don't override a newer state (e.g. user already became active).
        user = await db.get_user_by_entity(entity_id)
        if user and user.get("status") == "enriching":
            # First message may still fall back gracefully if profile is missing.
            await db.set_status(entity_id, "pending_first_message")


# ── Status poll ───────────────────────────────────────────────────────

@app.get("/api/status/{entity_id}")
async def get_status(entity_id: str):
    user = await db.get_user_by_entity(entity_id)
    return JSONResponse({"status": user.get("status", "unknown") if user else "not_found"})


@app.get("/api/chat/history/{entity_id}")
async def api_chat_history(entity_id: str):
    user = await db.get_user_by_entity(entity_id)
    if not user:
        raise HTTPException(404, "User not found")
    messages = await db.get_history_for_user(user)
    return JSONResponse({"messages": messages})


@app.post("/api/chat", response_model=ChatSendResponse)
async def api_chat(body: ChatSendRequest):
    user = await db.get_user_by_entity(body.entity_id)
    if not user:
        raise HTTPException(404, "User not found")

    text = (body.message or "").strip()
    if not text:
        raise HTTPException(422, "Message required")

    status = user.get("status")
    if status == "pending_auth":
        raise HTTPException(400, "Complete Google sign-in first.")

    if status == "enriching":
        raise HTTPException(409, "Enrichment in progress. Try again shortly.")

    if status == "pending_first_message":
        replies = await _first_message_web(user, text)
        return ChatSendResponse(replies=replies)

    if status == "active":
        profile = user.get("profile") or {}
        persona = profile.get("persona_description")
        facts = profile.get("facts_sheet")
        replies = await reply(text, persona, user=user, facts_sheet=facts)
        return ChatSendResponse(replies=replies)

    raise HTTPException(400, "Cannot chat in this state.")


async def _first_message_web(user: dict, user_text: str) -> list[str]:
    profile = user.get("profile") or {}
    first_name = profile.get("first_name") or user.get("name", "").split()[0] or "hey"
    persona = profile.get("persona_description") or ""
    facts = profile.get("facts_sheet")

    if persona:
        replies = await craft_first_message(first_name, persona, facts)
    else:
        replies = [f"hey {first_name} 👋 you're in. just talk to me."]

    await db.append_message_for_user(user, "user", user_text)
    for line in replies:
        await db.append_message_for_user(user, "assistant", line)
    await db.set_status(user["entity_id"], "active")
    return replies
