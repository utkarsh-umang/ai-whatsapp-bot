"""
Context enrichment pipeline.

Gmail OAuth gives us name + email.
GMAIL_SEARCH_PEOPLE gives us company domain.
We fetch social notification emails, run an LLM to extract structured account signals,
merge regex fallbacks, then call Perplexity Agent API (pro-search) for the full profile.
We classify a personality tier, build personality_brief, and persist EnrichmentPayload in MongoDB.
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
    PersonalityTier,
    SocialEmailInsights,
    SocialPlatformAccount,
    EnrichmentPayload,
    PerplexitySnapshot,
)

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_openai = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_SOCIAL_EXTRACT_MODEL = "gpt-4o-mini"


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

{social_email_block}Search for them using the identifiers above to avoid confusing them with someone else.
If the structured social-email section includes LinkedIn URLs or headlines, treat those as strong signals.
If it includes Instagram or other handles, use them to disambiguate and enrich persona.

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


def _format_social_insights_for_perplexity(insights: SocialEmailInsights) -> str:
    if not insights.summary and not insights.accounts:
        return ""
    lines = []
    if insights.summary:
        lines.append(f"Summary from notification emails: {insights.summary}")
    if insights.extraction_notes:
        lines.append(f"Extraction notes: {insights.extraction_notes}")
    lines.append(f"Confidence (email agent): {insights.confidence}")
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

    block = _format_social_insights_for_perplexity(social_insights)
    social_email_block = ""
    if block.strip():
        social_email_block = (
            "Structured data from their social network notification emails:\n"
            + block
            + "\n\n"
        )

    prompt = _PROFILE_PROMPT.format(
        available_info="\n".join(parts),
        social_email_block=social_email_block,
    )
    raw = await _call_perplexity_agent(prompt)
    excerpt = raw[:4000] + ("…" if len(raw) > 4000 else "")
    return _parse_json(raw), excerpt


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


def _sanitize_social_emails_for_llm(emails: list[dict], max_emails: int = 15) -> str:
    """Compact JSON for the extraction agent — truncate long bodies."""
    slim = []
    for e in emails[:max_emails]:
        body = (e.get("messageText") or e.get("body") or "") or ""
        if len(body) > 8000:
            body = body[:8000] + "…"
        slim.append({
            "sender": e.get("sender"),
            "subject": e.get("subject"),
            "preview": e.get("preview"),
            "messageText": body,
        })
    return json.dumps(slim, ensure_ascii=False)


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


def _insights_from_regex(handles: dict) -> SocialEmailInsights:
    accounts: list[SocialPlatformAccount] = []
    if handles.get("instagram"):
        accounts.append(SocialPlatformAccount(
            platform="instagram",
            handle=handles["instagram"],
        ))
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


def merge_regex_into_insights(insights: SocialEmailInsights, handles: dict) -> SocialEmailInsights:
    """Fill gaps from regex when the agent missed a handle or URL."""
    accounts = [a.model_copy(deep=True) for a in insights.accounts]

    def _has_ig() -> bool:
        return any(a.platform.lower() == "instagram" and a.handle for a in accounts)

    def _has_li_url() -> bool:
        return any(a.platform.lower() == "linkedin" and a.profile_url for a in accounts)

    if not _has_ig() and handles.get("instagram"):
        accounts.append(SocialPlatformAccount(
            platform="instagram",
            handle=handles["instagram"],
        ))
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


async def extract_social_email_insights(
    name: str,
    email: str,
    emails: list[dict],
) -> SocialEmailInsights:
    """
    LLM agent over social notification emails; falls back to regex-derived insights if needed.
    """
    if not emails:
        return SocialEmailInsights()

    regex_handles = extract_social_handles(emails)
    if not _openai:
        logging.warning("[enrichment] OPENAI_API_KEY missing — using regex-only social insights")
        return _insights_from_regex(regex_handles)

    user_block = f"User identity (from Gmail): name={name!r}, email={email!r}\n\nEmails JSON:\n"
    payload = user_block + _sanitize_social_emails_for_llm(emails)

    try:
        resp = await _openai.chat.completions.create(
            model=_SOCIAL_EXTRACT_MODEL,
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

    merged = merge_regex_into_insights(insights, regex_handles)
    return merged


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

    # Step 1: Social notification emails → LLM agent + regex merge → structured insights
    social_insights = SocialEmailInsights()
    try:
        social_emails = await fetch_social_platform_emails(entity_id)
        if social_emails:
            logging.info(f"[enrichment] first social email keys: {list(social_emails[0].keys())}")
            logging.debug(f"[enrichment] first social email: {social_emails[0]}")
        social_insights = await extract_social_email_insights(name, email, social_emails)
        logging.info(f"[enrichment] social insights: {social_insights.model_dump()}")
    except Exception as e:
        logging.error(f"[enrichment] social email pipeline failed: {e}", exc_info=True)

    # Step 2: Perplexity — web profile using identity + structured social insights
    raw_excerpt: str | None = None
    merged: dict = {}
    try:
        merged, raw_excerpt = await resolve_profile(
            name, email, company_domain, social_insights
        )
        logging.info(f"[enrichment] Perplexity result: {merged}")
    except Exception as e:
        logging.error(f"[enrichment] Perplexity failed: {e}", exc_info=True)

    # Patch LinkedIn URL from email insights if Perplexity omitted it
    if not merged.get("linkedin"):
        for acc in social_insights.accounts:
            if acc.platform.lower() == "linkedin" and acc.profile_url:
                merged["linkedin"] = acc.profile_url
                break

    perplexity_snapshot = PerplexitySnapshot(
        model_preset=_PRESET,
        fields={
            "name": merged.get("name"),
            "role": merged.get("role"),
            "company": merged.get("company"),
            "bio": merged.get("bio"),
            "linkedin": merged.get("linkedin"),
            "location": merged.get("location"),
        },
        raw_response_excerpt=raw_excerpt,
    )
    enrichment_payload = EnrichmentPayload(
        social_email_insights=social_insights,
        perplexity_snapshot=perplexity_snapshot,
    )

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
        enrichment=enrichment_payload,
    )
