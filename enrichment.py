"""
Context enrichment pipeline.

Flow:
  1. Gmail OAuth gives us name + email (+ inferred company domain from contacts).
  2. We pull 5 Gmail "buckets": social notifications, food delivery, travel/rides,
     shopping+subscriptions, and finance transaction alerts.
  3. Each bucket is parsed by a dedicated gpt-4o-mini extractor into strict JSON.
  4. Perplexity (pro-search) takes identity + every handle/identifier we found
     and returns a structured public profile (role, company, bio, linkedin, etc.).
  5. A persona-writer agent merges all structured signals into a single
     `persona_description` — the only thing chat prompts need to reference.

The result is a UserProfile whose `persona_description` is the single source of
truth for tone + grounded facts across first-message and ongoing replies.
"""

import asyncio
import os
import re
import json
import httpx
import logging
from collections import Counter
from composio import Composio
from openai import AsyncOpenAI
from models import (
    UserProfile,
    SocialEmailInsights,
    SocialPlatformAccount,
    FoodInsights,
    TravelInsights,
    ShoppingInsights,
    FinanceInsights,
    LifestyleInsights,
    EnrichmentPayload,
    PerplexitySnapshot,
    PersonaSnapshot,
)

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_openai = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

_EXTRACT_MODEL = "gpt-4o-mini"
_PERSONA_MODEL = "gpt-4o"


# ── Gmail identity ────────────────────────────────────────────────────

def _composio() -> Composio:
    return Composio(api_key=COMPOSIO_API_KEY)


def _execute(slug: str, arguments: dict, user_id: str) -> dict:
    """Synchronous Composio tool execution — always call via asyncio.to_thread."""
    result = _composio().tools.execute(
        slug,
        arguments,
        user_id=user_id,
        dangerously_skip_version_check=True,
    )
    if isinstance(result, dict):
        return result.get("data") or result
    return {}


async def get_gmail_identity(entity_id: str) -> tuple[str, str, str | None]:
    """Returns (name, email, company_domain | None)."""
    profile = await asyncio.to_thread(
        _execute, "GMAIL_GET_PROFILE", {}, entity_id
    )
    email: str = profile.get("emailAddress", "")
    name: str = profile.get("name") or _name_from_email(email)

    company_domain: str | None = None
    try:
        people = await asyncio.to_thread(
            _execute,
            "GMAIL_SEARCH_PEOPLE",
            {"query": name, "page_size": 20},
            entity_id,
        )
        company_domain = _infer_company_domain(email, people)
    except Exception:
        pass

    return name, email, company_domain


def _name_from_email(email: str) -> str:
    local = email.split("@")[0]
    parts = re.split(r"[._\-]", local)
    return " ".join(p.capitalize() for p in parts if p)


def _infer_company_domain(user_email: str, people_result: dict) -> str | None:
    PERSONAL = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "icloud.com", "me.com", "proton.me", "protonmail.com",
    }
    user_domain = user_email.split("@")[-1]
    if user_domain not in PERSONAL:
        return user_domain

    contacts = people_result.get("people", []) if isinstance(people_result, dict) else []
    domains = []
    for person in contacts:
        for e in person.get("emailAddresses", []):
            addr = e.get("value", "")
            if "@" in addr:
                d = addr.split("@")[-1]
                if d not in PERSONAL:
                    domains.append(d)

    if not domains:
        return None
    most_common, _ = Counter(domains).most_common(1)[0]
    return most_common


# ── Gmail buckets (India-first) ──────────────────────────────────────

# One-line Gmail search queries per bucket. Kept tight to avoid noise; the
# extractor LLM is the second filter. Bodies are truncated before being passed
# to any LLM in _slim_emails_for_llm.

BUCKET_QUERIES: dict[str, str] = {
    "social": (
        "from:mail.instagram.com OR from:facebookmail.com "
        "OR from:notification.instagram.com OR from:linkedin.com"
    ),
    "food": (
        "from:swiggy.in OR from:zomato.com OR from:ubereats.com "
        "OR subject:(\"order confirmed\" OR \"order placed\" OR \"order delivered\") "
    ),
    "travel": (
        "from:uber.com OR from:olacabs.com OR from:rapido.bike "
        "OR from:makemytrip.com OR from:goibibo.com OR from:cleartrip.com "
        "OR from:booking.com OR from:airbnb.com "
        "OR subject:(itinerary OR \"booking confirmed\" OR \"e-ticket\" OR \"trip receipt\") "
    ),
    "shopping": (
        "from:amazon.in OR from:flipkart.com OR from:myntra.com "
        "OR from:netflix.com OR from:spotify.com OR from:apple.com OR from:youtube.com "
        "OR subject:(\"your order\" OR \"subscription\" OR \"renewal\") "
    ),
    "finance": (
        "from:alerts@hdfcbank.net OR from:noreply@sbi.co.in OR from:credit_cards@icicibank.com "
        "OR from:alerts.axisbank@axisbank.com OR from:noreply@kotak.com "
        "OR subject:(\"transaction alert\" OR \"spent on your card\" OR \"debited\" OR \"credited\")"
    ),
}

BUCKET_FETCH_CAP: int = 25      # how many emails to fetch from Gmail per bucket
BUCKET_LLM_CAP: int = 12        # how many to pass into the extractor LLM
BODY_TRUNCATE_CHARS: int = 6000 # per-email body cap before LLM


async def fetch_bucket(entity_id: str, bucket: str, query: str, max_results: int = BUCKET_FETCH_CAP) -> list[dict]:
    """Generic wrapper over GMAIL_FETCH_EMAILS. Returns a list of email dicts."""
    logging.info(f"[enrichment] fetching bucket={bucket!r} for {entity_id} (max={max_results})")
    raw = await asyncio.to_thread(
        _execute,
        "GMAIL_FETCH_EMAILS",
        {"query": query, "max_results": max_results},
        entity_id,
    )
    if isinstance(raw, list):
        emails = raw
    elif isinstance(raw, dict):
        emails = raw.get("messages", []) or raw.get("emails", []) or ([raw] if raw else [])
    else:
        emails = []
    logging.info(f"[enrichment] bucket={bucket!r} fetched {len(emails)} emails")
    return emails


def _slim_emails_for_llm(emails: list[dict], max_emails: int = BUCKET_LLM_CAP) -> str:
    """Compact JSON for an extractor — truncate long bodies."""
    slim = []
    for e in emails[:max_emails]:
        body = (e.get("messageText") or e.get("body") or "") or ""
        if len(body) > BODY_TRUNCATE_CHARS:
            body = body[:BODY_TRUNCATE_CHARS] + "…"
        slim.append({
            "sender": e.get("sender"),
            "subject": e.get("subject"),
            "preview": e.get("preview"),
            "messageText": body,
        })
    return json.dumps(slim, ensure_ascii=False)


# ── Perplexity profile resolution (Agent API, same backend as browser Pro Search) ──

_AGENT_API_URL = "https://api.perplexity.ai/v1/agent"
# openai/gpt-5.1, web_search + fetch_url, up to 3 steps — matches browser "Pro Search"
_PRESET = "pro-search"

_AGENT_INSTRUCTIONS = (
    "Do not add any citation markers (e.g. [web:1], [page:2]) to your response. "
    "Return ONLY a valid JSON object. No markdown fences, no prose, no citations. "
    "If a field is genuinely not findable, use null. Never guess or hallucinate."
)

_PROFILE_PROMPT = """\
Build a professional profile for this person using their public online presence.

Known info (from Gmail + email signal extraction):
{available_info}

{handles_block}Use these identifiers to pin down the exact right person and avoid collisions.
LinkedIn URL/headline (when present) is the strongest single signal — anchor on it.
Instagram/Facebook/X handles can disambiguate further and hint at personality or interests.

Return ONLY this JSON shape:
{{
  "name": "full name as publicly known, or null",
  "role": "current job title, or null",
  "company": "company name, or null",
  "bio": "1-2 sentences: what they work on, what they've built — specific, not generic; or null",
  "linkedin": "canonical LinkedIn URL, or null",
  "location": "city/country, or null",
  "interests_public": ["short public-interest keywords inferred from bio/handles", "max 6"]
}}

Only fill a field if you have evidence. Never guess.
"""


async def _call_perplexity_agent(prompt: str) -> str:
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY is not set")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            _AGENT_API_URL,
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "preset": _PRESET,
                "input": prompt,
                "instructions": _AGENT_INSTRUCTIONS,
            },
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logging.error(f"Perplexity API HTTP Error: {e.response.status_code}")
            logging.error(f"Response Body: {e.response.text}")
            raise
        data = resp.json()

    text_parts: list[str] = []
    for item in data.get("output", []):
        for content_block in item.get("content", []):
            if content_block.get("type") == "output_text":
                text_parts.append(content_block.get("text", ""))

    return "".join(text_parts).strip()


def _parse_json(raw: str) -> dict:
    clean = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


def _format_handles_for_perplexity(insights: SocialEmailInsights) -> str:
    if not insights.summary and not insights.accounts:
        return ""
    lines = []
    if insights.summary:
        lines.append(f"Email-agent summary: {insights.summary}")
    if insights.extraction_notes:
        lines.append(f"Extraction notes: {insights.extraction_notes}")
    lines.append(f"Social email confidence: {insights.confidence}")
    for acc in insights.accounts:
        bits = [f"platform={acc.platform}"]
        if acc.handle:
            bits.append(f"handle={acc.handle}")
        if acc.profile_url:
            bits.append(f"url={acc.profile_url}")
        if acc.display_name:
            bits.append(f"name={acc.display_name}")
        if acc.headline:
            bits.append(f"headline={acc.headline}")
        if acc.snippets:
            bits.append("snippets=" + "; ".join(acc.snippets[:3]))
        lines.append("  • " + " | ".join(bits))
    return "\n".join(lines)


async def resolve_profile(
    name: str,
    email: str,
    company_domain: str | None,
    social_insights: SocialEmailInsights,
) -> tuple[dict, str]:
    """
    Calls Perplexity with Gmail identity + structured social-email insights.
    Returns (parsed_json_fields, raw_response_excerpt_for_storage).
    """
    parts = [f"Name: {name}", f"Email: {email}"]
    if company_domain:
        parts.append(f"Company domain: {company_domain}")

    handles = _format_handles_for_perplexity(social_insights)
    handles_block = ""
    if handles.strip():
        handles_block = (
            "Structured signals from their social network notification emails:\n"
            + handles
            + "\n\n"
        )

    prompt = _PROFILE_PROMPT.format(
        available_info="\n".join(parts),
        handles_block=handles_block,
    )
    raw = await _call_perplexity_agent(prompt)
    excerpt = raw[:4000] + ("…" if len(raw) > 4000 else "")
    return _parse_json(raw), excerpt


# ── Social bucket: LLM extractor + regex fallback ────────────────────

_SOCIAL_EXTRACT_SYSTEM = """\
You read social-network notification emails (Instagram, LinkedIn, Facebook, etc.) and extract \
ONLY facts that are explicitly present in the text.

Return ONLY valid JSON (no markdown fences) with this exact shape:
{
  "summary": null or a single sentence on what these emails reveal about the user,
  "accounts": [
    {
      "platform": "instagram|linkedin|facebook|x|threads|other",
      "handle": null or username without @,
      "profile_url": null or full https URL,
      "display_name": null or name as shown in the email,
      "headline": null or job title / bio line from LinkedIn footer or similar,
      "snippets": ["short useful quotes from the email, max 3 items"]
    }
  ],
  "confidence": "high|medium|low",
  "extraction_notes": null or brief caveats
}

Rules:
- If the email is not about this user (generic promo), leave accounts empty.
- Do not invent URLs or handles; null if unsure.
- LinkedIn footer often says: This email was intended for Name (Headline) — use that for headline.
"""


def _social_handles_regex(emails: list[dict]) -> dict[str, str | None]:
    """Regex fallback for Instagram/LinkedIn/Facebook signals."""
    instagram_handle: str | None = None
    linkedin_url: str | None = None
    linkedin_headline: str | None = None

    def _str(val) -> str:
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return json.dumps(val)
        return str(val) if val is not None else ""

    for email in emails:
        text = " ".join(filter(None, [
            _str(email.get("messageText")),
            _str(email.get("preview")),
            _str(email.get("subject")),
        ]))
        sender = _str(email.get("sender")).lower()

        if "instagram" in sender:
            if not instagram_handle:
                m = re.search(r'instagram\.com/([A-Za-z0-9._]+)(?:/|\?|$|\s)', text)
                if m and m.group(1) not in ("accounts", "p", "explore", "stories", "direct"):
                    instagram_handle = m.group(1)
                if not instagram_handle:
                    m = re.search(r'@([A-Za-z0-9._]{3,30})', text)
                    if m:
                        instagram_handle = m.group(1)

        if "linkedin" in sender:
            if not linkedin_url:
                m = re.search(r'linkedin\.com/in/([A-Za-z0-9\-]+)', text)
                if m:
                    linkedin_url = f"https://www.linkedin.com/in/{m.group(1)}"
            if not linkedin_headline:
                m = re.search(
                    r'[Tt]his email was intended for [^(]+?\(([^)]+)\)',
                    text,
                )
                if m:
                    linkedin_headline = m.group(1).strip()

        if instagram_handle and linkedin_url and linkedin_headline:
            break

    return {
        "instagram": instagram_handle,
        "linkedin_url": linkedin_url,
        "linkedin_headline": linkedin_headline,
    }


def _insights_from_regex(handles: dict) -> SocialEmailInsights:
    accounts: list[SocialPlatformAccount] = []
    if handles.get("instagram"):
        accounts.append(SocialPlatformAccount(platform="instagram", handle=handles["instagram"]))
    if handles.get("linkedin_url") or handles.get("linkedin_headline"):
        accounts.append(SocialPlatformAccount(
            platform="linkedin",
            profile_url=handles.get("linkedin_url"),
            headline=handles.get("linkedin_headline"),
        ))
    return SocialEmailInsights(
        summary=None,
        accounts=accounts,
        confidence="medium" if accounts else "low",
        extraction_notes="Regex-only extraction (no LLM).",
    )


def _merge_regex_into_social(insights: SocialEmailInsights, handles: dict) -> SocialEmailInsights:
    """Fill gaps from regex when the LLM missed a handle or URL."""
    accounts = [a.model_copy(deep=True) for a in insights.accounts]

    def _has_ig() -> bool:
        return any(a.platform.lower() == "instagram" and a.handle for a in accounts)

    def _has_li_url() -> bool:
        return any(a.platform.lower() == "linkedin" and a.profile_url for a in accounts)

    if not _has_ig() and handles.get("instagram"):
        accounts.append(SocialPlatformAccount(platform="instagram", handle=handles["instagram"]))
    if not _has_li_url() and handles.get("linkedin_url"):
        accounts.append(SocialPlatformAccount(
            platform="linkedin",
            profile_url=handles["linkedin_url"],
            headline=handles.get("linkedin_headline"),
        ))
    elif handles.get("linkedin_headline"):
        for a in accounts:
            if a.platform.lower() == "linkedin" and not a.headline:
                a.headline = handles["linkedin_headline"]
                break

    return insights.model_copy(update={"accounts": accounts})


async def extract_social_insights(name: str, email: str, emails: list[dict]) -> SocialEmailInsights:
    """LLM agent over social notification emails; regex fallback fills gaps."""
    if not emails:
        return SocialEmailInsights()

    regex_handles = _social_handles_regex(emails)
    if not _openai:
        logging.warning("[enrichment] OPENAI_API_KEY missing — using regex-only social insights")
        return _insights_from_regex(regex_handles)

    user_block = f"User identity (from Gmail): name={name!r}, email={email!r}\n\nEmails JSON:\n"
    payload = user_block + _slim_emails_for_llm(emails)

    try:
        resp = await _openai.chat.completions.create(
            model=_EXTRACT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SOCIAL_EXTRACT_SYSTEM},
                {"role": "user", "content": payload},
            ],
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        insights = SocialEmailInsights.model_validate(data)
    except Exception as e:
        logging.error(f"[enrichment] social email LLM extract failed: {e}", exc_info=True)
        insights = SocialEmailInsights()

    return _merge_regex_into_social(insights, regex_handles)


# ── Lifestyle bucket extractors ──────────────────────────────────────

_FOOD_EXTRACT_SYSTEM = """\
You read food-delivery order emails (Swiggy, Zomato, Uber Eats, etc.) and extract ONLY facts \
explicitly present in the text.

Return ONLY valid JSON with this exact shape:
{
  "summary": null or one sentence describing this user's food habits,
  "cuisines": ["short cuisine/dish tags, max 6"],
  "favourite_restaurants": ["restaurant names mentioned more than once, max 5"],
  "typical_order_times": ["late night|morning|lunch|dinner|weekend — inferred only from timestamps in text"],
  "cities": ["delivery city or neighbourhood names, max 3"],
  "order_count_seen": null or integer count of distinct orders visible,
  "confidence": "high|medium|low",
  "extraction_notes": null or brief caveats
}

Rules:
- Only include items you can see in the emails. Never invent restaurants or cuisines.
- Skip promotional / discount-only emails; only count real orders.
"""

_TRAVEL_EXTRACT_SYSTEM = """\
You read ride and travel emails (Uber, Ola, Rapido, MakeMyTrip, Goibibo, Booking.com, Airbnb, airlines, hotels) \
and extract ONLY facts explicitly present in the text.

Return ONLY valid JSON with this exact shape:
{
  "summary": null or one sentence on their travel/ride pattern,
  "ride_apps": ["uber|ola|rapido|other"],
  "cities": ["cities/areas used for rides, max 4"],
  "travel_destinations": ["cities or countries from bookings, max 6"],
  "travel_modes": ["flight|train|bus|hotel|airbnb"],
  "recent_trip_months": ["YYYY-MM, max 6"],
  "confidence": "high|medium|low",
  "extraction_notes": null or brief caveats
}

Rules:
- Only use real confirmations/receipts. Ignore promo and price-drop alerts.
- Never invent dates or destinations.
"""

_SHOPPING_EXTRACT_SYSTEM = """\
You read shopping and subscription emails (Amazon.in, Flipkart, Myntra, Netflix, Spotify, Apple, YouTube etc.) \
and extract ONLY facts explicitly present in the text.

Return ONLY valid JSON with this exact shape:
{
  "summary": null or one sentence describing shopping/subscription behaviour,
  "shopping_platforms": ["amazon|flipkart|myntra|..."],
  "purchase_categories": ["electronics|books|fashion|home|kitchen|..."],
  "subscriptions": ["netflix|spotify|youtube premium|apple one|..."],
  "notable_items": ["specific products mentioned, max 5"],
  "confidence": "high|medium|low",
  "extraction_notes": null or brief caveats
}

Rules:
- Only include items visible in the emails. Never invent products.
- Do not list every item — surface the notable ones a human would find interesting.
"""

_FINANCE_EXTRACT_SYSTEM = """\
You read bank transaction-alert emails and extract ONLY coarse patterns. \
NEVER capture amounts, card numbers, or account numbers.

Return ONLY valid JSON with this exact shape:
{
  "summary": null or one sentence on their spend pattern at a category level,
  "spend_categories": ["travel|food|shopping|subscriptions|fuel|utilities|other"],
  "frequent_merchants": ["merchant names that appear more than once, max 5"],
  "confidence": "high|medium|low",
  "extraction_notes": null or brief caveats
}

Hard rules:
- Do NOT output amounts, currencies, balances, account numbers, or card numbers.
- Only merchant names and high-level category tags. Nothing else.
"""


async def _run_bucket_extractor(
    system_prompt: str,
    name: str,
    email: str,
    emails: list[dict],
    model_cls,
):
    """Run a single bucket extractor. Returns a pydantic instance (default on failure)."""
    if not emails:
        return model_cls()
    if not _openai:
        logging.warning(f"[enrichment] OPENAI_API_KEY missing — skipping {model_cls.__name__}")
        return model_cls()

    user_block = f"User identity (from Gmail): name={name!r}, email={email!r}\n\nEmails JSON:\n"
    payload = user_block + _slim_emails_for_llm(emails)

    try:
        resp = await _openai.chat.completions.create(
            model=_EXTRACT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload},
            ],
            max_tokens=1500,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        return model_cls.model_validate(data)
    except Exception as e:
        logging.error(f"[enrichment] {model_cls.__name__} extract failed: {e}", exc_info=True)
        return model_cls()


async def extract_food_insights(name: str, email: str, emails: list[dict]) -> FoodInsights:
    return await _run_bucket_extractor(_FOOD_EXTRACT_SYSTEM, name, email, emails, FoodInsights)


async def extract_travel_insights(name: str, email: str, emails: list[dict]) -> TravelInsights:
    return await _run_bucket_extractor(_TRAVEL_EXTRACT_SYSTEM, name, email, emails, TravelInsights)


async def extract_shopping_insights(name: str, email: str, emails: list[dict]) -> ShoppingInsights:
    return await _run_bucket_extractor(_SHOPPING_EXTRACT_SYSTEM, name, email, emails, ShoppingInsights)


async def extract_finance_insights(name: str, email: str, emails: list[dict]) -> FinanceInsights:
    return await _run_bucket_extractor(_FINANCE_EXTRACT_SYSTEM, name, email, emails, FinanceInsights)


# ── Persona writer agent ─────────────────────────────────────────────

_PERSONA_WRITER_SYSTEM = """\
You are a persona writer. Given structured signals about a user (all grounded in their Gmail + \
public web), write a compact, durable description of who they are. This description will be \
injected into every future chat as the model's memory of this person, alongside a separate \
deterministic facts sheet — so your job is to capture IDENTITY and TONE, not enumerate every fact.

Output format — plain text, no JSON, no markdown fences, no headings. Use short labelled lines, \
in this order, separated by single newlines:

Name: <full name or first name only if that's all we have>
Role: <current job title at company — only if grounded; otherwise: unknown>
Bio: <one sentence, specific, grounded in the web/profile data; otherwise: unknown>
Location: <city/country if known; otherwise: unknown>
Online presence: <public handles and one-line impression, e.g. "IG @xyz; LinkedIn /in/abc">
Vibe: <one short sentence that captures who they feel like based on all signals — e.g. "Bengaluru-based backend engineer who lives on late-night biryani and weekend Goa trips". Use VERBATIM specifics from the input (restaurant names, cuisines, destinations, subscriptions) when present. If signals are thin, say "unclear — calibrate from their messages".>
Unknowns: <comma-separated list of fields we couldn't confirm (e.g. "role, company, city") so the chat model never bluffs>
Tone: <one line of voice guidance, e.g. "witty, peer-to-peer, concise; mirror their register">

Hard rules:
- Only restate facts present in the input JSON. If a signal is low-confidence or missing, say "unknown".
- In the Vibe line, prefer verbatim specifics from the input (cuisine names, restaurant names, city names, destinations, subscriptions) over vague adjectives. Do NOT say "occasional traveller" if you have actual destinations — say the destinations.
- Never output amounts, card numbers, or bank names. Treat finance signals as coarse lifestyle context only (e.g. "frequent traveller", "regular subscriber") and never cite merchants.
- Never invent restaurants, cities, job titles, or projects.
- Keep the whole output under ~150 words. Tight and usable.
"""


async def write_persona_description(
    identity: dict,
    social: SocialEmailInsights,
    lifestyle: LifestyleInsights,
    perplexity_fields: dict,
) -> tuple[str | None, str | None]:
    """Returns (persona_description, raw_excerpt_for_audit)."""
    if not _openai:
        logging.warning("[enrichment] OPENAI_API_KEY missing — skipping persona writer")
        return None, None

    payload = {
        "identity": identity,
        "perplexity": perplexity_fields,
        "social": social.model_dump(),
        "lifestyle": lifestyle.model_dump(),
    }

    try:
        resp = await _openai.chat.completions.create(
            model=_PERSONA_MODEL,
            max_tokens=600,
            messages=[
                {"role": "system", "content": _PERSONA_WRITER_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return None, None
        excerpt = raw[:4000] + ("…" if len(raw) > 4000 else "")
        return raw, excerpt
    except Exception as e:
        logging.error(f"[enrichment] persona writer failed: {e}", exc_info=True)
        return None, None


# ── Deterministic facts sheet ────────────────────────────────────────
#
# Built without an LLM, directly from structured enrichment objects, so no
# hallucination risk. The chat model is instructed to pick at most ONE item
# from this sheet per message. Finance data is intentionally excluded — it
# only contributes at most a coarse "frequent traveller" / "regular subscriber"
# hint via the persona writer, never a line here.

_FACTS_SHEET_MAX_ITEMS_PER_FIELD = 4


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        if not raw:
            continue
        s = str(raw).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _cap(items: list[str], n: int = _FACTS_SHEET_MAX_ITEMS_PER_FIELD) -> list[str]:
    return _dedup_preserve_order(items)[:n]


def build_facts_sheet(
    *,
    name: str,
    perplexity_fields: dict,
    social: SocialEmailInsights,
    lifestyle: LifestyleInsights,
) -> str:
    """Return a compact bullet list of concrete, grounded facts. Empty lines skipped.

    Shape (only lines that have data are emitted):
      - Professional: <role> at <company> — <bio>
      - Location: <city/country>
      - Online: IG @handle; LinkedIn /in/xyz; LinkedIn headline: "..."
      - Food: loves <cuisines>; frequent at <restaurants>; orders at <times>; in <cities>
      - Travel: been to <destinations>; uses <ride apps>; modes: <modes>
      - Shopping: shops on <platforms>; buys <categories>; notable items: <items>
      - Subscriptions: <subs>
      - Interests: <public interests>

    Finance is intentionally excluded. No amounts, banks, merchants, or cards.
    """
    lines: list[str] = []

    # Professional
    role = (perplexity_fields or {}).get("role")
    company = (perplexity_fields or {}).get("company")
    bio = (perplexity_fields or {}).get("bio")
    prof_bits: list[str] = []
    if role and company:
        prof_bits.append(f"{role} at {company}")
    elif role:
        prof_bits.append(str(role))
    elif company:
        prof_bits.append(f"at {company}")
    if bio:
        prof_bits.append(str(bio).strip().rstrip("."))
    if prof_bits:
        lines.append("- Professional: " + " — ".join(prof_bits))

    # Location
    location = (perplexity_fields or {}).get("location")
    if location:
        lines.append(f"- Location: {location}")

    # Online presence
    online_bits: list[str] = []
    ig_handle: str | None = None
    linkedin_url: str | None = (perplexity_fields or {}).get("linkedin")
    linkedin_headline: str | None = None
    for acc in social.accounts:
        platform = (acc.platform or "").lower()
        if platform == "instagram" and acc.handle and not ig_handle:
            ig_handle = acc.handle
        if platform == "linkedin":
            if acc.profile_url and not linkedin_url:
                linkedin_url = acc.profile_url
            if acc.headline and not linkedin_headline:
                linkedin_headline = acc.headline
    if ig_handle:
        online_bits.append(f"IG @{ig_handle.lstrip('@')}")
    if linkedin_url:
        online_bits.append(f"LinkedIn {linkedin_url}")
    if linkedin_headline:
        online_bits.append(f'LinkedIn headline: "{linkedin_headline}"')
    if online_bits:
        lines.append("- Online: " + "; ".join(online_bits))

    # Food
    food = lifestyle.food
    food_bits: list[str] = []
    cuisines = _cap(food.cuisines)
    restaurants = _cap(food.favourite_restaurants)
    times = _cap(food.typical_order_times, 3)
    cities = _cap(food.cities, 2)
    if cuisines:
        food_bits.append("loves " + ", ".join(cuisines))
    if restaurants:
        food_bits.append("frequent at " + ", ".join(restaurants))
    if times:
        food_bits.append("orders " + ", ".join(times))
    if cities:
        food_bits.append("in " + ", ".join(cities))
    if food_bits:
        lines.append("- Food: " + "; ".join(food_bits))

    # Travel
    travel = lifestyle.travel
    travel_bits: list[str] = []
    destinations = _cap(travel.travel_destinations, 5)
    ride_apps = _cap(travel.ride_apps, 3)
    modes = _cap(travel.travel_modes, 3)
    travel_cities = _cap(travel.cities, 3)
    if destinations:
        travel_bits.append("been to " + ", ".join(destinations))
    if travel_cities:
        travel_bits.append("rides in " + ", ".join(travel_cities))
    if ride_apps:
        travel_bits.append("uses " + ", ".join(ride_apps))
    if modes:
        travel_bits.append("modes: " + ", ".join(modes))
    if travel_bits:
        lines.append("- Travel: " + "; ".join(travel_bits))

    # Shopping (items) + Subscriptions (split so the chat model can pick cleanly)
    shopping = lifestyle.shopping
    shop_bits: list[str] = []
    platforms = _cap(shopping.shopping_platforms, 3)
    categories = _cap(shopping.purchase_categories, 4)
    items = _cap(shopping.notable_items, 4)
    if platforms:
        shop_bits.append("shops on " + ", ".join(platforms))
    if categories:
        shop_bits.append("buys " + ", ".join(categories))
    if items:
        shop_bits.append("notable items: " + ", ".join(items))
    if shop_bits:
        lines.append("- Shopping: " + "; ".join(shop_bits))

    subs = _cap(shopping.subscriptions, 5)
    if subs:
        lines.append("- Subscriptions: " + ", ".join(subs))

    # Public interests (from Perplexity)
    interests_raw = (perplexity_fields or {}).get("interests_public") or []
    if isinstance(interests_raw, str):
        # Perplexity snapshot serialises this as JSON string; tolerate both.
        try:
            parsed = json.loads(interests_raw)
            if isinstance(parsed, list):
                interests_raw = parsed
            else:
                interests_raw = []
        except Exception:
            interests_raw = []
    interests = _cap([str(i) for i in interests_raw], 6)
    if interests:
        lines.append("- Interests: " + ", ".join(interests))

    if not lines:
        return ""

    header = (
        "Concrete facts about this person (reference at MOST one per message, "
        "only when it naturally fits; never mention banks, money, or cards):"
    )
    return header + "\n" + "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────

async def run_enrichment(entity_id: str) -> UserProfile:
    """
    Full pipeline. Returns a UserProfile with persona_description populated.
    Call this after Gmail OAuth completes.
    """
    logging.info(f"[enrichment] starting pipeline for entity={entity_id}")
    name, email, company_domain = await get_gmail_identity(entity_id)
    logging.info(f"[enrichment] identity: name={name!r} email={email!r} domain={company_domain!r}")
    first_name = name.split()[0].capitalize() if name else "there"

    # Step 1: fetch all Gmail buckets in parallel.
    bucket_names = ["social", "food", "travel", "shopping", "finance"]
    fetch_tasks = [
        fetch_bucket(entity_id, b, BUCKET_QUERIES[b]) for b in bucket_names
    ]
    fetched = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    bucket_emails: dict[str, list[dict]] = {}
    for b, result in zip(bucket_names, fetched):
        if isinstance(result, Exception):
            logging.error(f"[enrichment] bucket={b} fetch failed: {result}")
            bucket_emails[b] = []
        else:
            bucket_emails[b] = result

    # Step 2: run every extractor in parallel.
    social_task = extract_social_insights(name, email, bucket_emails["social"])
    food_task = extract_food_insights(name, email, bucket_emails["food"])
    travel_task = extract_travel_insights(name, email, bucket_emails["travel"])
    shopping_task = extract_shopping_insights(name, email, bucket_emails["shopping"])
    finance_task = extract_finance_insights(name, email, bucket_emails["finance"])

    social_insights, food, travel, shopping, finance = await asyncio.gather(
        social_task, food_task, travel_task, shopping_task, finance_task,
        return_exceptions=False,
    )
    logging.info(f"[enrichment] social insights confidence={social_insights.confidence} accounts={len(social_insights.accounts)}")
    logging.info(f"[enrichment] food confidence={food.confidence} cuisines={food.cuisines[:3]}")
    logging.info(f"[enrichment] travel confidence={travel.confidence} destinations={travel.travel_destinations[:3]}")
    logging.info(f"[enrichment] shopping confidence={shopping.confidence} platforms={shopping.shopping_platforms[:3]}")
    logging.info(f"[enrichment] finance confidence={finance.confidence} categories={finance.spend_categories[:3]}")

    lifestyle = LifestyleInsights(
        food=food, travel=travel, shopping=shopping, finance=finance
    )

    # Step 3: Perplexity — web profile using identity + all social handles.
    perplexity_excerpt: str | None = None
    perplexity_fields: dict = {}
    try:
        perplexity_fields, perplexity_excerpt = await resolve_profile(
            name, email, company_domain, social_insights
        )
        logging.info(f"[enrichment] Perplexity result keys: {list(perplexity_fields.keys())}")
    except Exception as e:
        logging.error(f"[enrichment] Perplexity failed: {e}", exc_info=True)

    # Patch LinkedIn URL from email insights if Perplexity omitted it.
    if not perplexity_fields.get("linkedin"):
        for acc in social_insights.accounts:
            if acc.platform.lower() == "linkedin" and acc.profile_url:
                perplexity_fields["linkedin"] = acc.profile_url
                break

    perplexity_snapshot = PerplexitySnapshot(
        model_preset=_PRESET,
        fields={
            "name": perplexity_fields.get("name"),
            "role": perplexity_fields.get("role"),
            "company": perplexity_fields.get("company"),
            "bio": perplexity_fields.get("bio"),
            "linkedin": perplexity_fields.get("linkedin"),
            "location": perplexity_fields.get("location"),
            "interests_public": json.dumps(perplexity_fields.get("interests_public") or []),
        },
        raw_response_excerpt=perplexity_excerpt,
    )

    # Step 4: persona writer — merge everything into a single persona_description.
    identity = {
        "name": name,
        "email": email,
        "company_domain": company_domain,
    }
    persona_description, persona_excerpt = await write_persona_description(
        identity, social_insights, lifestyle, perplexity_fields
    )
    if persona_description:
        logging.info(f"[enrichment] persona_description written ({len(persona_description)} chars)")
    else:
        logging.warning("[enrichment] persona_description could not be written")

    persona_snapshot = PersonaSnapshot(
        model=_PERSONA_MODEL,
        persona_description=persona_description,
        raw_response_excerpt=persona_excerpt,
    )

    # Step 5: deterministic facts sheet — concrete items, privacy-filtered.
    facts_sheet = build_facts_sheet(
        name=name,
        perplexity_fields=perplexity_fields,
        social=social_insights,
        lifestyle=lifestyle,
    )
    if facts_sheet:
        line_count = facts_sheet.count("\n- ")
        logging.info(f"[enrichment] facts_sheet built ({line_count} items, {len(facts_sheet)} chars)")
    else:
        logging.info("[enrichment] facts_sheet empty (no grounded items)")

    enrichment_payload = EnrichmentPayload(
        social_email_insights=social_insights,
        lifestyle_insights=lifestyle,
        perplexity_snapshot=perplexity_snapshot,
        persona_snapshot=persona_snapshot,
    )

    return UserProfile(
        name=perplexity_fields.get("name") or name,
        email=email,
        first_name=first_name,
        role=perplexity_fields.get("role"),
        company=perplexity_fields.get("company"),
        bio=perplexity_fields.get("bio"),
        linkedin=perplexity_fields.get("linkedin"),
        location=perplexity_fields.get("location"),
        persona_description=persona_description,
        facts_sheet=facts_sheet or None,
        enrichment=enrichment_payload,
    )
