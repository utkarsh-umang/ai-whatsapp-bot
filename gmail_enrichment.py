"""
Gmail Context enrichment pipeline.

This module accesses the user's Google account via Composio to extract
recent emails and parse their role/company context using GPT-4o.
This avoids needing Web Search / Perplexity entirely.
"""

import asyncio
import os
import re
import json
import logging
from composio import Composio
from openai import AsyncOpenAI
from models import UserProfile, PersonalityTier

# We reuse the logic for tier/brief from original enrichment
from enrichment import classify_tier, build_personality_brief

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

def _composio() -> Composio:
    return Composio(api_key=COMPOSIO_API_KEY)

def _execute(slug: str, arguments: dict, user_id: str) -> dict:
    """Synchronous Composio tool execution — avoid blocking."""
    result = _composio().tools.execute(
        slug,
        arguments,
        user_id=user_id,
        dangerously_skip_version_check=True,
    )
    if isinstance(result, dict):
        return result.get("data") or result
    return {}

async def get_gmail_identity(entity_id: str) -> tuple[str, str]:
    """Returns (name, email)."""
    try:
        profile = await asyncio.to_thread(
            _execute, "GMAIL_GET_PROFILE", {}, entity_id
        )
    except Exception as e:
        logging.error(f"Error fetching profile: {e}")
        profile = {}
    
    email: str = profile.get("emailAddress", "")
    name: str = profile.get("name") or _name_from_email(email)

    return name, email

def _name_from_email(email: str) -> str:
    local = email.split("@")[0]
    parts = re.split(r"[._\-]", local)
    return " ".join(p.capitalize() for p in parts if p)

async def fetch_recent_emails(entity_id: str) -> list[dict]:
    """Fetch the recent emails via Composio to build context."""
    res = await asyncio.to_thread(_execute, "GMAIL_FETCH_EMAILS", {}, entity_id)
    
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        return res.get("messages", []) or res.get("emails", []) or [res]
    
    return []


_PROFILE_EXTRACTION_PROMPT = """
You are an expert executive assistant. I am passing you recent emails involving {user_name} ({user_email}).
Analyze these emails, especially sent emails, signatures, and recent technical/business context, and determine their professional profile.

Return a strict JSON object with this shape:
{{
  "role": "their job title or role, null if unknown",
  "company": "the company they work for, null if unknown",
  "bio": "A 1-2 sentence description of what they specifically work on or what is keeping them busy based on the emails."
}}

Rules:
- If the emails do not explicitly provide enough context, infer what you safely can.
- Use null if missing. Never hallucinate titles.
- DO NOT wrap the output in markdown codeblocks. Return pure JSON.
"""

async def analyze_emails_with_llm(name: str, email: str, emails_data: list) -> dict:
    if not emails_data:
        return {}

    # Just pass up to 10 latest emails to avoid overwhelming the token limit
    top_emails = emails_data[:10]
    emails_text = json.dumps(top_emails)

    if len(emails_text) > 60000:
        emails_text = emails_text[:60000]

    prompt = _PROFILE_EXTRACTION_PROMPT.format(user_name=name, user_email=email)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Here are the recent emails:\n{emails_text}"}
    ]

    try:
        resp = await _client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=300
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        logging.error(f"Failed to extract profile with LLM: {e}")
        return {}

async def run_enrichment(entity_id: str) -> UserProfile:
    """
    Full pipeline. Returns a UserProfile with personality_brief populated.
    Called immediately after Gmail OAuth.
    """
    name, email = await get_gmail_identity(entity_id)
    first_name = name.split()[0].capitalize() if name else "there"

    try:
        emails_data = await fetch_recent_emails(entity_id)
    except Exception as e:
        logging.error(f"Failed to fetch recent emails: {e}")
        emails_data = []

    profile_data = await analyze_emails_with_llm(name, email, emails_data)

    merged = {
        "name": name,
        "role": profile_data.get("role"),
        "company": profile_data.get("company"),
        "bio": profile_data.get("bio"),
        "linkedin": None, # Unused via Gmail enrichment
        "location": None
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
