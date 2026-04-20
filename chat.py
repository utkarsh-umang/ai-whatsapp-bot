"""
The chatbot.

Two modes:
  craft_first_message()  — called once after enrichment. The wow moment.
  reply()                — all subsequent messages. Loads history, calls GPT,
                           saves both turns to MongoDB.

Tone + grounded facts come from a single `persona_description` string produced
by the persona-writer agent in enrichment.py. No tier/brief branching.
"""

import os
import json
from openai import AsyncOpenAI
import db
from typing import Any

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL_FIRST  = "gpt-4o"       # better creative writing for the wow moment
MODEL_CHAT   = "gpt-4o-mini"  # fast + cheap for ongoing replies

_MULTI_MESSAGE_SYSTEM = """\
Output format:
- Return ONLY valid JSON (no markdown fences, no prose).
- Shape: {"messages": ["...","..."]} where each item is one WhatsApp-style line.

Rules:
- Each message must be ONE line, ideally ONE sentence.
- Default to 1-3 messages.
- If the user asked a question, message[0] must answer it directly.
- No bullet points. No numbered lists. No long paragraphs.
"""


def _normalize_lines(raw_lines: list[Any] | None, *, max_lines: int = 3) -> list[str]:
    if not raw_lines:
        return []
    out: list[str] = []
    for item in raw_lines:
        if not isinstance(item, str):
            continue
        s = item.replace("\r", "").strip()
        if not s:
            continue
        # Force single-line bubbles.
        s = " ".join(s.splitlines()).strip()
        if s:
            out.append(s)
        if len(out) >= max_lines:
            break
    return out


# ── Base system prompt ───────────────────────────────────────────────
#
# faff's core personality is fixed. The persona_description from enrichment
# supplies the specific facts about who they're talking to.

_BASE_SYSTEM = """\
You are faff. You live in WhatsApp. You are witty, brief, peer-to-peer — never an assistant trying to help.

Tone rules:
- Short. 1-3 sentences. WhatsApp, not email.
- Dry wit over enthusiasm. Never say "Great question!", "Certainly!", or "How can I help you today?".
- No bullet points for conversational replies.
- Mirror their register — if they're terse, be terse; if they're playful, match it.
- Reference what you know about them naturally — never as an announcement ("I see you…").
- Use at most one specific grounded detail per message. Don't stack every fact you have.

Hard rules (privacy + honesty):
- Never invent facts. If you don't know something about them, just don't mention it.
- Never mention bank names, card numbers, or money amounts.
- Don't quote the persona description back at them.
"""


def _build_system(persona_description: str | None, facts_sheet: str | None = None) -> str:
    base = _BASE_SYSTEM
    if persona_description:
        base += (
            "\nWho you're talking to (identity + tone — grounded from their Gmail + public web):\n"
            + persona_description.strip()
            + "\n"
        )
    if facts_sheet:
        base += "\n" + facts_sheet.strip() + "\n"
    base += (
        "\nCritical: You already know this person from when they signed up. "
        "You're not meeting them for the first time on every message. "
        "Refer back to what you know only when it adds something."
    )
    return base.strip()


# ── First message ─────────────────────────────────────────────────────

_FIRST_MESSAGE_PROMPT = """\
Write the very first message faff sends to {first_name} after they sign up. This is the WOW moment.

Identity + tone:
{persona_description}

{facts_block}Rules:
- Open with their first name and a casual greeting (hey, not "Hello").
- Pick the SINGLE most surprising or telling concrete item from the facts above and lead with it. Use the VERBATIM specifics (restaurant names, cities, destinations, subscriptions) — don't generalise them away.
- Specific beats generic: "saw you're building agentic stuff at WordsWorth" beats "saw you work in AI"; "hope Goa was worth the 4 AM flight" beats "hope your last trip was fun".
- Only ONE concrete callback. Do not stack multiple facts.
- Keep it SHORT. 3-4 short lines max. WhatsApp, not email.
- End with one casual line about what faff does — invite them to just talk.
- Do NOT use bullet points. Do NOT say "I've connected your accounts." Do NOT mention banks, money, cards, or finance alerts. Do NOT quote the persona back at them.
- If the facts sheet is empty or the persona has "Unknowns", stay warm and generic — never bluff.

Write just the messages. No quotes, no explanation.
"""


async def craft_first_message(
    first_name: str,
    persona_description: str,
    facts_sheet: str | None = None,
) -> list[str]:
    facts_block = ""
    if facts_sheet and facts_sheet.strip():
        facts_block = facts_sheet.strip() + "\n\n"
    prompt = _FIRST_MESSAGE_PROMPT.format(
        first_name=first_name,
        persona_description=persona_description or "unknown — keep it warm and generic.",
        facts_block=facts_block,
    )
    resp = await _client.chat.completions.create(
        model=MODEL_FIRST,
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _MULTI_MESSAGE_SYSTEM + "\nDefault to 2-4 messages for the very first hello."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}

    lines = _normalize_lines((data or {}).get("messages"), max_lines=4)
    if not lines:
        lines = _normalize_lines(raw.split("\n"), max_lines=4)
    return lines


# ── Ongoing replies ───────────────────────────────────────────────────

async def reply(
    user_message: str,
    persona_description: str | None,
    *,
    user: dict[str, Any],
    facts_sheet: str | None = None,
) -> list[str]:
    """Load conversation history, call GPT, save both turns, return response."""
    history = await db.get_history_for_user(user)

    messages = [
        {"role": "system", "content": _build_system(persona_description, facts_sheet)},
        {"role": "system", "content": _MULTI_MESSAGE_SYSTEM},
    ]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    resp = await _client.chat.completions.create(
        model=MODEL_CHAT,
        response_format={"type": "json_object"},
        max_tokens=500,
        messages=messages,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    lines = _normalize_lines((data or {}).get("messages"), max_lines=3)
    if not lines:
        lines = _normalize_lines([raw], max_lines=1) or ["something broke on my end, try again?"]

    await db.append_message_for_user(user, "user", user_message)
    for line in lines:
        await db.append_message_for_user(user, "assistant", line)

    return lines
