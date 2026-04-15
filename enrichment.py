"""
Context enrichment pipeline.

Gmail OAuth gives us name + email.
GMAIL_SEARCH_PEOPLE gives us company domain.
Perplexity resolves the full professional profile.
We then classify a personality tier and build a personality_brief
that gets injected into every system prompt for this user.
"""

import os
import re
import json
import httpx
from collections import Counter
from composio_langchain import ComposioToolSet, Action
from models import UserProfile, PersonalityTier

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


# ── Gmail identity ────────────────────────────────────────────────────

async def get_gmail_identity(entity_id: str) -> tuple[str, str, str | None]:
    """Returns (name, email, company_domain | None)."""
    toolset = ComposioToolSet(api_key=COMPOSIO_API_KEY)

    profile = toolset.execute_action(
        action=Action.GMAIL_GET_PROFILE,
        params={},
        entity_id=entity_id,
    )
    email: str = profile.get("emailAddress", "")
    name: str = profile.get("name") or _name_from_email(email)

    company_domain: str | None = None
    try:
        people = toolset.execute_action(
            action=Action.GMAIL_SEARCH_PEOPLE,
            params={"query": name, "page_size": 20},
            entity_id=entity_id,
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


# ── Perplexity profile resolution ────────────────────────────────────

_AGENT_INSTRUCTIONS = (
    "Do not add citation markers. "
    "Return ONLY a valid JSON object. No markdown fences, no prose."
)

_PROFILE_PROMPT = """\
Build a professional profile for this person using their online presence.

Known info:
{available_info}

Search for them using name + company/domain to avoid confusing them with someone else.

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


def _call_perplexity(prompt: str) -> str:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            "https://api.perplexity.ai/v1/agent",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "preset": "pro-search",
                "input": prompt,
                "instructions": _AGENT_INSTRUCTIONS,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    parts = []
    for item in data.get("output", []):
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                parts.append(block.get("text", ""))
    return "".join(parts).strip()


def _parse_json(raw: str) -> dict:
    clean = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


def resolve_profile(name: str, email: str, company_domain: str | None) -> dict:
    parts = [f"Name: {name}", f"Email: {email}"]
    if company_domain:
        parts.append(f"Company domain: {company_domain}")
    prompt = _PROFILE_PROMPT.format(available_info="\n".join(parts))
    raw = _call_perplexity(prompt)
    return _parse_json(raw)


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
    name, email, company_domain = await get_gmail_identity(entity_id)
    first_name = name.split()[0].capitalize() if name else "there"

    perplexity_data = resolve_profile(name, email, company_domain)

    # Merge — Perplexity data preferred, Gmail as fallback
    merged = {
        "name": perplexity_data.get("name") or name,
        "role": perplexity_data.get("role"),
        "company": perplexity_data.get("company"),
        "bio": perplexity_data.get("bio"),
        "linkedin": perplexity_data.get("linkedin"),
        "location": perplexity_data.get("location"),
    }

    tier = classify_tier(merged["role"], merged["company"], merged["bio"])
    brief = build_personality_brief(merged, tier)

    return UserProfile(
        name=merged["name"],
        email=email,
        first_name=first_name,
        role=merged["role"],
        company=merged["company"],
        bio=merged["bio"],
        linkedin=merged["linkedin"],
        location=merged["location"],
        personality_tier=tier,
        personality_brief=brief,
    )
