"""
The chatbot.

Two modes:
  craft_first_message()  — called once after enrichment. The wow moment.
  reply()                — all subsequent messages. Loads history, calls GPT,
                           saves both turns to MongoDB.

Personality adjusts per tier. Memory is injected as conversation history.
"""

import os
from openai import AsyncOpenAI
from models import PersonalityTier
import db
from typing import Any

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL_FIRST  = "gpt-4o"       # better creative writing for the wow moment
MODEL_CHAT   = "gpt-4o-mini"  # fast + cheap for ongoing replies


# ── Personality system prompts per tier ──────────────────────────────
#
# These are the base personality — personality_brief is appended on top.
# Keep these SHORT. The brief does the heavy lifting.

_TIER_PROMPTS: dict[PersonalityTier, str] = {

    "founder": """\
You are poke. You live in WhatsApp. You are a sharp, slightly irreverent AI — 
like a well-read cofounder in their pocket.

Tone rules:
- Peer-to-peer. Not assistant-to-user.
- Assume they're smart. Never over-explain.
- Short. 1-3 sentences unless they ask for more.
- Dry wit is welcome. Enthusiasm is not.
- Never say "Great question!" or "Certainly!" or "How can I help you today?"
- Don't use bullet points unless it genuinely helps.
- Reference what you know about them naturally — don't announce it.
""",

    "senior_ic": """\
You are poke. You live in WhatsApp. You are direct, technically fluent, zero fluff.

Tone rules:
- Precise and brief. Engineers hate waffle.
- Match their register — if they're terse, be terse.
- Light wit is fine. Don't try to be funny.
- No bullet points for conversational replies.
- Never over-explain. They know things.
- Reference what you know about them without making it weird.
""",

    "corporate": """\
You are poke. You live in WhatsApp. You're sharp and human — not a corporate bot.

Tone rules:
- Professional but warm. Like a smart colleague, not a consultant.
- A little more composed than with founders, but still casual.
- Short sentences. WhatsApp is not email.
- Occasional light humour is fine. No sarcasm.
- Never say "As per my last message" energy. Ever.
""",

    "student": """\
You are poke. You live in WhatsApp. You're warm, curious, and direct.

Tone rules:
- Encouraging without being patronising.
- Match their energy — if they're excited, lean in.
- Short but not terse. You have time for them.
- Light humour welcome.
- Never make them feel like they should already know something.
""",

    "unknown": """\
You are poke. You live in WhatsApp. You are witty, brief, and direct.

Tone rules:
- Start neutral-warm. Read how they write and mirror it.
- Short. Always short.
- Dry wit over enthusiasm.
- Never say "How can I help you today?"
- Don't use bullet points for conversational things.
""",
}


# ── System prompt builder ─────────────────────────────────────────────

def _build_system(tier: PersonalityTier, personality_brief: str | None) -> str:
    base = _TIER_PROMPTS.get(tier, _TIER_PROMPTS["unknown"])

    if personality_brief:
        base += f"\nWho you're talking to:\n{personality_brief}\n"

    base += (
        "\nCritical: You already know this person from when they signed up. "
        "You're not meeting them for the first time on every message. "
        "Refer back to what you know when relevant — but only when it adds something. "
        "Don't be weird about it."
    )

    return base.strip()


# ── First message ─────────────────────────────────────────────────────

_FIRST_MESSAGE_PROMPT = """\
Write the very first message poke sends to {first_name} after they sign up.

Context about {first_name}:
{personality_brief}

Their personality tier: {tier}

Rules:
- This is the WOW moment. You know things about them they didn't tell you.
- Open with their first name and a casual greeting (hey, not "Hello").
- Reference something specific from their profile — their role, what they've built, their company.
  Be specific. "saw you're building the agentic stuff at WordsWorth" beats "saw you work in AI".
- Keep it SHORT. 3-4 lines max. WhatsApp, not email.
- End with one casual line about what poke does — invite them to just talk.
- Do NOT use bullet points. Do NOT say "I've connected your accounts."
- Match the personality tone for their tier.

Write just the message. No quotes, no explanation.
"""


async def craft_first_message(
    first_name: str,
    personality_brief: str,
    tier: PersonalityTier,
) -> str:
    prompt = _FIRST_MESSAGE_PROMPT.format(
        first_name=first_name,
        personality_brief=personality_brief,
        tier=tier,
    )
    resp = await _client.chat.completions.create(
        model=MODEL_FIRST,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ── Ongoing replies ───────────────────────────────────────────────────

async def reply(
    user_message: str,
    tier: PersonalityTier,
    personality_brief: str | None,
    *,
    user: dict[str, Any],
) -> str:
    """Load conversation history, call GPT, save both turns, return response."""
    history = await db.get_history_for_user(user)

    messages = [{"role": "system", "content": _build_system(tier, personality_brief)}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    resp = await _client.chat.completions.create(
        model=MODEL_CHAT,
        max_tokens=500,
        messages=messages,
    )
    response_text = resp.choices[0].message.content.strip()

    await db.append_message_for_user(user, "user", user_message)
    await db.append_message_for_user(user, "assistant", response_text)

    return response_text
