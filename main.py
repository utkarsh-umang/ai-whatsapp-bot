import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from composio import Composio

import db
import periskope
from models import (
    SetupPhoneRequest,
    SetupPhoneResponse,
    PeriskopeWebhook,
    ChatSendRequest,
    ChatSendResponse,
)
from enrichment import run_enrichment
from chat import craft_first_message, reply

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
# Create this in Composio dashboard → Auth Configs → Gmail → copy the config ID
COMPOSIO_GMAIL_AUTH_CONFIG_ID = os.getenv("COMPOSIO_GMAIL_AUTH_CONFIG_ID")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
# This is the bot's phone number used for the "Click to Chat" link
BOT_PHONE = os.getenv("PERISKOPE_PHONE", "919XXXXXXXXX")


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
    """Gmail OAuth is done. Fire enrichment in background; user continues to hi + chat."""
    await db.set_status(entity_id, "pending_first_message")

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


# ── Phone setup ───────────────────────────────────────────────────────

@app.post("/setup/phone", response_model=SetupPhoneResponse)
async def setup_phone(body: SetupPhoneRequest):
    user = await db.get_user_by_entity(body.entity_id)
    if not user:
        raise HTTPException(404, "Complete Google auth first.")

    phone = body.phone.replace(" ", "").replace("-", "").lstrip("+")
    if not phone.isdigit() or len(phone) < 10:
        raise HTTPException(422, "Invalid phone number.")

    if len(phone) == 10:
        phone = "91" + phone

    await db.upsert_user(body.entity_id, {
        "phone": phone,
        "status": "pending_first_message",
    })

    return SetupPhoneResponse(wa_link=f"https://wa.me/{BOT_PHONE}?text=Hi")


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

    if status == "pending_first_message":
        reply_text = await _first_message_web(user, text)
        return ChatSendResponse(reply=reply_text)

    if status == "active":
        profile = user.get("profile") or {}
        tier = profile.get("personality_tier") or "unknown"
        brief = profile.get("personality_brief")
        reply_text = await reply(text, tier, brief, user=user)
        return ChatSendResponse(reply=reply_text)

    raise HTTPException(400, "Cannot chat in this state.")


# ── Periskope webhook ─────────────────────────────────────────────────

@app.post("/webhook/periskope")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    # Only handle incoming messages — ignore delivery acks, reactions, etc.
    event = payload.get("event")
    if event != "message.created":
        return JSONResponse({"ok": True})

    msg = PeriskopeWebhook(**payload)
    if not msg.data:
        return JSONResponse({"ok": True})

    # from_me=True means the BOT sent this — skip to avoid echo loops
    if msg.data.from_me:
        return JSONResponse({"ok": True})

    phone = msg.data.sender()   # normalised, no @c.us
    body = msg.data.body

    if not phone or not body:
        return JSONResponse({"ok": True})

    user = await db.get_user_by_phone(phone)
    if not user:
        return JSONResponse({"ok": True})

    status = user.get("status")

    # ── First message → send the personalised welcome ─────────────────
    if status == "pending_first_message":
        await db.set_status(user["entity_id"], "active")
        background_tasks.add_task(_send_first_message, user, phone)
        return JSONResponse({"ok": True})

    # ── Active → normal chat ──────────────────────────────────────────
    if status == "active":
        background_tasks.add_task(_chat, user, body)
        return JSONResponse({"ok": True})

    return JSONResponse({"ok": True})


async def _send_first_message(user: dict, phone: str):
    profile = user.get("profile") or {}
    first_name = profile.get("first_name") or user.get("name", "").split()[0] or "hey"
    personality_brief = profile.get("personality_brief") or ""
    tier = profile.get("personality_tier") or "unknown"

    if personality_brief:
        # Full wow moment — we know who they are
        message = await craft_first_message(first_name, personality_brief, tier)
    else:
        # Enrichment failed or is still running — graceful fallback
        message = f"hey {first_name} 👋 you're in. just talk to me."

    await periskope.send(phone, message)

    # Save to history so future turns have context of the first message
    await db.append_message_for_user(user, "assistant", message)


async def _chat(user: dict, message: str):
    profile = user.get("profile") or {}
    tier = profile.get("personality_tier") or "unknown"
    brief = profile.get("personality_brief")
    phone = user.get("phone")
    if not phone:
        return

    try:
        response = await reply(message, tier, brief, user=user)
        await periskope.send(phone, response)
    except Exception:
        await periskope.send(phone, "something broke on my end, try again?")


async def _first_message_web(user: dict, user_text: str) -> str:
    profile = user.get("profile") or {}
    first_name = profile.get("first_name") or user.get("name", "").split()[0] or "hey"
    personality_brief = profile.get("personality_brief") or ""
    tier = profile.get("personality_tier") or "unknown"

    if personality_brief:
        message = await craft_first_message(first_name, personality_brief, tier)
    else:
        message = f"hey {first_name} 👋 you're in. just talk to me."

    await db.append_message_for_user(user, "user", user_text)
    await db.append_message_for_user(user, "assistant", message)
    await db.set_status(user["entity_id"], "active")
    return message
