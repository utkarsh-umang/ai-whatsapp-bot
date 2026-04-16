from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


PersonalityTier = Literal[
    "founder",      # early-stage, builder, startup
    "senior_ic",    # staff/principal/lead engineer, deep IC
    "corporate",    # big-co, enterprise, MBA-ish titles
    "student",      # intern, student, recent grad
    "unknown",      # couldn't determine
]


# ── Enrichment payloads (stored under UserProfile.enrichment) ─────────
# Versioned JSON so Mongo can evolve without breaking readers.

EnrichmentSchemaVersion = Literal[1]


class SocialPlatformAccount(BaseModel):
    """One surfaced account or signal from a social notification email."""

    platform: str  # instagram | linkedin | facebook | x | threads | other
    handle: Optional[str] = None
    profile_url: Optional[str] = None
    display_name: Optional[str] = None
    headline: Optional[str] = None
    snippets: list[str] = Field(default_factory=list)


class SocialEmailInsights(BaseModel):
    """Structured output from the social-email extraction agent."""

    summary: Optional[str] = None
    accounts: list[SocialPlatformAccount] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    extraction_notes: Optional[str] = None


class PerplexitySnapshot(BaseModel):
    """What we asked Perplexity for and what it returned (for audit / debugging)."""

    model_preset: str = "pro-search"
    fields: dict[str, Optional[str]] = Field(default_factory=dict)
    raw_response_excerpt: Optional[str] = None


class EnrichmentPayload(BaseModel):
    """
    Full enrichment record stored in MongoDB alongside flat UserProfile fields.
    """

    schema_version: EnrichmentSchemaVersion = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    social_email_insights: SocialEmailInsights
    perplexity_snapshot: Optional[PerplexitySnapshot] = None


class UserProfile(BaseModel):
    name: str
    email: str
    first_name: str
    role: Optional[str] = None
    company: Optional[str] = None
    bio: Optional[str] = None
    linkedin: Optional[str] = None
    location: Optional[str] = None
    personality_tier: PersonalityTier = "unknown"

    # The pre-built brief that gets injected into every system prompt
    personality_brief: Optional[str] = None

    # Rich structured enrichment (social agent + Perplexity); stored in MongoDB as JSON
    enrichment: Optional[EnrichmentPayload] = None


class User(BaseModel):
    entity_id: str                          # Composio entity ID (= onboarding session)
    phone: Optional[str] = None            # WA number with country code, no +
    email: Optional[str] = None
    name: Optional[str] = None
    status: str = "pending_auth"           # pending_auth → pending_first_message → active (optional pending_phone via /setup/phone)
    profile: Optional[UserProfile] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    ts: datetime = Field(default_factory=datetime.utcnow)


# ── API models ────────────────────────────────────────────────────────

class SetupPhoneRequest(BaseModel):
    entity_id: str
    phone: str


class SetupPhoneResponse(BaseModel):
    wa_link: str


class ChatSendRequest(BaseModel):
    entity_id: str
    message: str


class ChatSendResponse(BaseModel):
    reply: str


# ── Periskope webhook ─────────────────────────────────────────────────
#
# Periskope wraps all events in an outer envelope:
#   { "event": "message.created", "data": { ...message... }, "org_id": "...", "timestamp": "..." }
#
# Key fields inside data:
#   data.body         — the message text
#   data.from         — sender phone with @c.us suffix e.g. "919876543210@c.us"
#   data.sender_phone — same as from (alternative field)
#   data.from_me      — true when the BOT sent this — use to filter echoes
#   data.chat_id      — use as the target when replying

class PeriskopeMessageData(BaseModel):
    body: Optional[str] = None
    from_: Optional[str] = Field(None, alias="from")
    sender_phone: Optional[str] = None
    from_me: Optional[bool] = False
    chat_id: Optional[str] = None

    def sender(self) -> Optional[str]:
        """Normalised sender phone — strips @c.us/@g.us and leading +"""
        raw = self.sender_phone or self.from_
        if not raw:
            return None
        return raw.replace("@c.us", "").replace("@g.us", "").lstrip("+").strip()

    model_config = {"populate_by_name": True}


class PeriskopeWebhook(BaseModel):
    event: Optional[str] = None
    data: Optional[PeriskopeMessageData] = None
    org_id: Optional[str] = None
    timestamp: Optional[str] = None
