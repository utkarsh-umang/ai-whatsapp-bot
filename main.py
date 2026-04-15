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
from models import SetupPhoneRequest, SetupPhoneResponse, PeriskopeWebhook
from enrichment import run_enrichment
from chat import craft_first_message, reply

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
# Create this in Composio dashboard → Auth Configs → Gmail → copy the config ID
COMPOSIO_GMAIL_AUTH_CONFIG_ID = os.getenv("COMPOSIO_GMAIL_AUTH_CONFIG_ID")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
BOT_PHONE = os.getenv("BOT_PHONE", "919XXXXXXXXX")


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
    """Gmail OAuth is done. Fire enrichment in background, send to step 2."""
    await db.set_status(entity_id, "pending_phone")

    # Start enrichment immediately — it'll be done by the time they enter
    # their phone and text the bot (typically 15-25 seconds total)
    background_tasks.add_task(_run_enrichment_task, entity_id)

    return RedirectResponse(f"/?step=2&entity_id={entity_id}")


async def _run_enrichment_task(entity_id: str):
    """Runs enrichment, stores profile. Silently fails — user won't know."""
    try:
        profile = await run_enrichment(entity_id)
        await db.save_profile(entity_id, profile)
    except Exception as e:
        # Store a minimal profile so the bot still works
        import logging
        logging.error(f"Enrichment failed for {entity_id}: {e}")


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
        background_tasks.add_task(_chat, user, body, phone)
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
    await db.append_message(phone, "assistant", message)


async def _chat(user: dict, message: str, phone: str):
    profile = user.get("profile") or {}
    tier = profile.get("personality_tier") or "unknown"
    brief = profile.get("personality_brief")

    try:
        response = await reply(phone, message, tier, brief)
        await periskope.send(phone, response)
    except Exception:
        await periskope.send(phone, "something broke on my end, try again?")
