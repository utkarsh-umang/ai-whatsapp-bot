from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# ── Enrichment payloads (stored under UserProfile.enrichment) ─────────
# Versioned JSON so Mongo can evolve without breaking readers.

EnrichmentSchemaVersion = Literal[1, 2]


# ── Social (Instagram / LinkedIn / Facebook notification emails) ─────

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


# ── Lifestyle buckets (food, travel, shopping/subs, finance) ─────────

class FoodInsights(BaseModel):
    """Food-delivery signal from Swiggy / Zomato / Uber Eats etc."""

    summary: Optional[str] = None
    cuisines: list[str] = Field(default_factory=list)
    favourite_restaurants: list[str] = Field(default_factory=list)
    typical_order_times: list[str] = Field(default_factory=list)   # e.g. "late night", "lunch"
    cities: list[str] = Field(default_factory=list)                # delivery cities/areas
    order_count_seen: Optional[int] = None
    confidence: Literal["high", "medium", "low"] = "low"
    extraction_notes: Optional[str] = None


class TravelInsights(BaseModel):
    """Travel / rides signal from Uber, Ola, MMT, airlines, hotels."""

    summary: Optional[str] = None
    ride_apps: list[str] = Field(default_factory=list)             # uber, ola, rapido
    cities: list[str] = Field(default_factory=list)                # ride / travel cities
    travel_destinations: list[str] = Field(default_factory=list)   # cities/countries from trips
    travel_modes: list[str] = Field(default_factory=list)          # flight, train, bus, hotel
    recent_trip_months: list[str] = Field(default_factory=list)    # e.g. "2024-12"
    confidence: Literal["high", "medium", "low"] = "low"
    extraction_notes: Optional[str] = None


class ShoppingInsights(BaseModel):
    """Shopping + subscription signal from Amazon, Flipkart, Netflix, Spotify etc."""

    summary: Optional[str] = None
    shopping_platforms: list[str] = Field(default_factory=list)
    purchase_categories: list[str] = Field(default_factory=list)   # electronics, books, fashion
    subscriptions: list[str] = Field(default_factory=list)         # netflix, spotify, apple
    notable_items: list[str] = Field(default_factory=list)         # specific products if mentioned
    confidence: Literal["high", "medium", "low"] = "low"
    extraction_notes: Optional[str] = None


class FinanceInsights(BaseModel):
    """Coarse finance signal from bank transaction-alert emails.

    Intentionally COARSE — we never capture amounts or account numbers,
    only merchant categories and high-level spend patterns.
    """

    summary: Optional[str] = None
    spend_categories: list[str] = Field(default_factory=list)      # travel, food, shopping
    frequent_merchants: list[str] = Field(default_factory=list)    # merchant names only
    confidence: Literal["high", "medium", "low"] = "low"
    extraction_notes: Optional[str] = None


class LifestyleInsights(BaseModel):
    """Aggregate of all lifestyle buckets."""

    food: FoodInsights = Field(default_factory=FoodInsights)
    travel: TravelInsights = Field(default_factory=TravelInsights)
    shopping: ShoppingInsights = Field(default_factory=ShoppingInsights)
    finance: FinanceInsights = Field(default_factory=FinanceInsights)


# ── Perplexity + persona-writer snapshots ────────────────────────────

class PerplexitySnapshot(BaseModel):
    """What we asked Perplexity for and what it returned (for audit / debugging)."""

    model_preset: str = "pro-search"
    fields: dict[str, Optional[str]] = Field(default_factory=dict)
    raw_response_excerpt: Optional[str] = None


class PersonaSnapshot(BaseModel):
    """Raw output of the persona-writer agent, for audit / debugging."""

    model: str = "gpt-4o"
    persona_description: Optional[str] = None
    raw_response_excerpt: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EnrichmentPayload(BaseModel):
    """
    Full enrichment record stored in MongoDB alongside flat UserProfile fields.
    """

    schema_version: EnrichmentSchemaVersion = 2
    created_at: datetime = Field(default_factory=datetime.utcnow)
    social_email_insights: SocialEmailInsights = Field(default_factory=SocialEmailInsights)
    lifestyle_insights: LifestyleInsights = Field(default_factory=LifestyleInsights)
    perplexity_snapshot: Optional[PerplexitySnapshot] = None
    persona_snapshot: Optional[PersonaSnapshot] = None


class UserProfile(BaseModel):
    name: str
    email: str
    first_name: str
    role: Optional[str] = None
    company: Optional[str] = None
    bio: Optional[str] = None
    linkedin: Optional[str] = None
    location: Optional[str] = None

    # Single source of truth for tone/content in chat prompts.
    persona_description: Optional[str] = None

    # Rich structured enrichment (all buckets + Perplexity + persona); stored in MongoDB as JSON.
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
    replies: list[str]


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
