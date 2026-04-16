"""
Context enrichment pipeline.

Gmail OAuth gives us name + email.
GMAIL_SEARCH_PEOPLE gives us company domain.
Perplexity Agent API (pro-search) resolves the full professional profile.
We then classify a personality tier and build a personality_brief
that gets injected into every system prompt for this user.
"""

import asyncio
import os
import re
import json
import httpx
import logging
from collections import Counter
from composio import Composio
from models import UserProfile, PersonalityTier

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


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
    # ToolExecutionResponse is dict-like; unwrap common response shapes
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
    # If user is on a company domain, return it directly
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


# ── Perplexity profile resolution (Agent API, same backend as browser Pro Search) ──

_AGENT_API_URL = "https://api.perplexity.ai/v1/agent"
# openai/gpt-5.1, web_search + fetch_url, up to 3 steps — matches browser "Pro Search"
_PRESET = "pro-search"

_AGENT_INSTRUCTIONS = (
    "Do not add any citation markers (e.g. [web:1], [page:2]) to your response. "
    "Return ONLY a valid JSON object. No markdown fences, no prose, no citations."
)

_PROFILE_PROMPT = """\
Build a professional profile for this person using their online presence.

Known info:
{available_info}

{social_hint}Search for them using the identifiers above to avoid confusing them with someone else.
If a LinkedIn URL is provided, prioritise that as the primary source.
If an Instagram handle is provided, use it as a secondary signal for their public persona.

Return ONLY this JSON shape:
{{
  "name": "full name as publicly known",
  "role": "current job title",
  "company": "company name",
  "bio": "1-2 sentences: what they work on, what they've built — specific not generic",
  "linkedin": "URL or null",
  "location": "city/country or null"
}}

If a field is genuinely not findable, use null. Never guess or hallucinate.
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


async def resolve_profile(
    name: str,
    email: str,
    company_domain: str | None,
    social_handles: dict[str, str | None] | None = None,
) -> dict:
    parts = [f"Name: {name}", f"Email: {email}"]
    if company_domain:
        parts.append(f"Company domain: {company_domain}")

    social_hint = ""
    handles = social_handles or {}

    if handles.get("linkedin_headline"):
        parts.append(f"LinkedIn headline: {handles['linkedin_headline']}")
        social_hint = (
            "The person's own LinkedIn headline has been extracted directly from a "
            "LinkedIn notification email sent to them — treat it as ground truth for "
            "their current role and company. Use it to anchor your search. "
        )
    if handles.get("linkedin_url"):
        parts.append(f"LinkedIn profile: {handles['linkedin_url']}")
        social_hint += "A LinkedIn profile URL is also available — use it as the primary source. "
    if handles.get("instagram"):
        parts.append(f"Instagram handle: @{handles['instagram']}")
        if not handles.get("linkedin_headline") and not handles.get("linkedin_url"):
            social_hint = "An Instagram handle has been provided — use it to cross-reference their public persona. "

    prompt = _PROFILE_PROMPT.format(
        available_info="\n".join(parts),
        social_hint=social_hint,
    )
    raw = await _call_perplexity_agent(prompt)
    return _parse_json(raw)


# ── Social platform email pipeline ───────────────────────────────────

_SOCIAL_EMAIL_QUERY = (
    "from:mail.instagram.com OR from:facebookmail.com "
    "OR from:notification.instagram.com OR from:linkedin.com"
)

async def fetch_social_platform_emails(entity_id: str) -> list[dict]:
    """Search Gmail for Instagram and LinkedIn notification emails."""
    logging.info(f"[enrichment] fetching social platform emails for {entity_id}")
    raw = await asyncio.to_thread(
        _execute,
        "GMAIL_FETCH_EMAILS",
        {"query": _SOCIAL_EMAIL_QUERY, "max_results": 20},
        entity_id,
    )
    logging.info(f"[enrichment] GMAIL_FETCH_EMAILS raw type={type(raw).__name__} keys={list(raw.keys()) if isinstance(raw, dict) else 'n/a'}")
    logging.debug(f"[enrichment] GMAIL_FETCH_EMAILS raw={raw}")

    if isinstance(raw, list):
        logging.info(f"[enrichment] got {len(raw)} social emails (list)")
        return raw
    if isinstance(raw, dict):
        emails = raw.get("messages", []) or raw.get("emails", []) or [raw]
        logging.info(f"[enrichment] got {len(emails)} social emails (dict)")
        return emails
    logging.warning(f"[enrichment] unexpected GMAIL_FETCH_EMAILS response type: {type(raw)}")
    return []


def extract_social_handles(emails: list[dict]) -> dict[str, str | None]:
    """
    Parse Instagram/LinkedIn platform emails to extract the user's own
    social signals. Returns:
      - instagram: handle | None
      - linkedin_url: profile URL | None
      - linkedin_headline: the user's own LinkedIn headline from the
          "This email was intended for Name (Headline)" footer | None
    """
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
        sender = _str(email.get("sender"))
        logging.debug(f"[enrichment] email sender={sender!r} text_len={len(text)}")

        if "instagram" in sender.lower():
            if not instagram_handle:
                # Instagram profile URLs in email body
                m = re.search(r'instagram\.com/([A-Za-z0-9._]+)(?:/|\?|$|\s)', text)
                if m and m.group(1) not in ("accounts", "p", "explore", "stories", "direct"):
                    instagram_handle = m.group(1)
                # Fallback: "@username" mention in body
                if not instagram_handle:
                    m = re.search(r'@([A-Za-z0-9._]{3,30})', text)
                    if m:
                        instagram_handle = m.group(1)

        if "linkedin" in sender.lower():
            # Profile URL (not always present)
            if not linkedin_url:
                m = re.search(r'linkedin\.com/in/([A-Za-z0-9\-]+)', text)
                if m:
                    linkedin_url = f"https://www.linkedin.com/in/{m.group(1)}"

            # LinkedIn footer: "This email was intended for Name (Headline)"
            # This appears in every LinkedIn notification email and contains
            # the user's own current headline.
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



# ── Personality tier classification ──────────────────────────────────

def classify_tier(role: str | None, company: str | None, bio: str | None) -> PersonalityTier:
    """
    Classify the user into a personality tier based on their profile.
    This shapes how the chatbot talks to them.
    """
    text = " ".join(filter(None, [role, company, bio])).lower()

    FOUNDER_SIGNALS = [
        "founder", "co-founder", "ceo", "cto", "founding engineer",
        "founding", "building", "0 to 1", "early stage", "startup", "stealth"
    ]
    SENIOR_IC_SIGNALS = [
        "staff", "principal", "senior", "lead engineer", "architect",
        "distinguished", "fellow", "vp of engineering", "engineering manager",
        "head of", "director of engineering",
    ]
    CORPORATE_SIGNALS = [
        "analyst", "associate", "manager", "consultant", "enterprise",
        "fortune", "accenture", "deloitte", "mckinsey", "banking",
        "investment", "finance", "ibm", "oracle", "sap",
    ]
    STUDENT_SIGNALS = [
        "student", "intern", "undergraduate", "graduate student",
        "phd", "msc", "mba student", "bootcamp", "recent grad",
    ]

    if any(s in text for s in FOUNDER_SIGNALS):
        return "founder"
    if any(s in text for s in SENIOR_IC_SIGNALS):
        return "senior_ic"
    if any(s in text for s in STUDENT_SIGNALS):
        return "student"
    if any(s in text for s in CORPORATE_SIGNALS):
        return "corporate"
    return "unknown"


# ── Personality brief builder ─────────────────────────────────────────

def build_personality_brief(profile: dict, tier: PersonalityTier) -> str:
    """
    Returns a compact string injected into every system prompt.
    Tells the agent exactly who they're talking to.
    """
    lines = []

    if profile.get("name"):
        lines.append(f"Name: {profile['name']}")
    if profile.get("role") and profile.get("company"):
        lines.append(f"Role: {profile['role']} at {profile['company']}")
    elif profile.get("role"):
        lines.append(f"Role: {profile['role']}")
    if profile.get("bio"):
        lines.append(f"What they do: {profile['bio']}")
    if profile.get("location"):
        lines.append(f"Location: {profile['location']}")

    tier_notes = {
        "founder": "builder mindset, bias to action, speaks in shipped things not titles",
        "senior_ic": "technically deep, no hand-holding needed, values precision",
        "corporate": "professional context, still human, appreciates clarity",
        "student": "early in career, curious, enthusiasm > credentials",
        "unknown": "vibe unclear — read their messages and calibrate",
    }
    lines.append(f"Vibe: {tier_notes[tier]}")

    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────

async def run_enrichment(entity_id: str) -> UserProfile:
    """
    Full pipeline. Returns a UserProfile with personality_brief populated.
    Call this after Gmail OAuth completes.
    """
    logging.info(f"[enrichment] starting pipeline for entity={entity_id}")
    name, email, company_domain = await get_gmail_identity(entity_id)
    logging.info(f"[enrichment] identity: name={name!r} email={email!r} domain={company_domain!r}")
    first_name = name.split()[0].capitalize() if name else "there"

    # Step 1: Find Instagram/LinkedIn notification emails → extract handles
    try:
        social_emails = await fetch_social_platform_emails(entity_id)
        # Log first email structure so we can verify field names
        if social_emails:
            logging.info(f"[enrichment] first social email keys: {list(social_emails[0].keys())}")
            logging.debug(f"[enrichment] first social email: {social_emails[0]}")
        social_handles = extract_social_handles(social_emails)
        logging.info(f"[enrichment] extracted handles: {social_handles}")
    except Exception as e:
        logging.error(f"[enrichment] social email fetch/extract failed: {e}", exc_info=True)
        social_handles = {"instagram": None, "linkedin_url": None, "linkedin_headline": None}

    # Step 2: Perplexity search — now enriched with social handles
    logging.info(f"[enrichment] calling Perplexity with handles={social_handles}")
    try:
        merged = await resolve_profile(name, email, company_domain, social_handles)
        logging.info(f"[enrichment] Perplexity result: {merged}")
    except Exception as e:
        logging.error(f"[enrichment] Perplexity failed: {e}", exc_info=True)
        merged = {}

    # Patch in the LinkedIn URL we extracted if Perplexity didn't find one
    if not merged.get("linkedin") and social_handles.get("linkedin_url"):
        merged["linkedin"] = social_handles["linkedin_url"]

    tier = classify_tier(merged.get("role"), merged.get("company"), merged.get("bio"))
    brief = build_personality_brief(merged, tier)

    return UserProfile(
        name=merged.get("name") or name,
        email=email,
        first_name=first_name,
        role=merged.get("role"),
        company=merged.get("company"),
        bio=merged.get("bio"),
        linkedin=merged.get("linkedin"),
        location=merged.get("location"),
        personality_tier=tier,
        personality_brief=brief,
    )
