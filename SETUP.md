# Local setup

This guide walks you through running **faff** on your own machine. It assumes no prior setup — you’ll create the accounts, grab the API keys, and start the app end-to-end.

> End result: `http://localhost:8080` running the web app, with working Google sign-in (via Composio), enrichment (OpenAI + Perplexity), and a local MongoDB.

---

## 1. Prerequisites

Install these before anything else:

- **Docker Desktop** — easiest way to run the app + MongoDB together. [Download](https://www.docker.com/products/docker-desktop/).
  - Confirm it works: `docker --version` and `docker compose version` both print something.
- **Git** — to clone the repo.
- **ngrok** (or any HTTPS tunnel) — the Gmail OAuth flow redirects back to a public HTTPS URL, so `http://localhost` alone won’t work end-to-end. [Install ngrok](https://ngrok.com/download) and sign up for the free tier.

You do **not** need to install Python, Node, or MongoDB locally — Docker handles everything.

---

## 2. Clone the repo

```bash
git clone <this-repo-url> faff
cd faff
cp .env.example .env
```

Open `.env` in your editor. You’ll fill it in over the next few sections. It looks like:

```
COMPOSIO_API_KEY=
COMPOSIO_GMAIL_AUTH_CONFIG_ID=
PERPLEXITY_API_KEY=
OPENAI_API_KEY=
MONGODB_URL=mongodb://localhost:27047
BASE_URL=https://your-ngrok-url.ngrok.io
```

Leave `MONGODB_URL` as-is — `docker compose` overrides it internally to point at the bundled Mongo container, and nothing else needs it.

---

## 3. Get an OpenAI API key

Used for structured email extraction, persona writing, and chat replies.

1. Paste it into `.env`:

```
OPENAI_API_KEY=sk-...
```

The app uses OpenAI’s standard chat/completions endpoints — no special org or project setup is required.

---

## 4. Get a Perplexity API key

Used to resolve a user’s public profile (role, company, LinkedIn, bio, etc.) from their name + email domain.

1. Paste it into `.env`:

```
PERPLEXITY_API_KEY=pplx-...
```

---

## 5. Set up Composio (Gmail auth)

Composio handles Google OAuth for you — you do **not** need to create a Google Cloud project or OAuth client yourself.

### 5a. Create a Composio account and get the API key

1. Go to [app.composio.dev](https://app.composio.dev/) and sign up.
2. In the dashboard, open **Developers → API Keys** (or the “API Keys” section in your profile).
3. Create a new API key and copy it.
4. Paste into `.env`:

```
COMPOSIO_API_KEY=...
```

### 5b. Create a Gmail Auth Config

This is the per-integration config the code references via `COMPOSIO_GMAIL_AUTH_CONFIG_ID`. The repo ships a one-off script that creates it for you and prints the ID — no dashboard clicks needed.

**Option 1 (recommended): use the helper script**

Make sure `COMPOSIO_API_KEY` is set in `.env` (from step 5a), then from the project root:

```bash
pip install composio python-dotenv
python3 scripts/create_gmail_auth_config.py
```

It will print a single line — the auth config ID (looks like `ac_...`). Paste it into `.env`:

```
COMPOSIO_GMAIL_AUTH_CONFIG_ID=ac_...
```

The script creates the config with `type: "use_composio_managed_auth"`, so you don’t need a Google Cloud project or your own OAuth client. You only need to run it **once** per Composio account; re-running creates a new config each time.

If you don’t want to install deps locally, you can also run the script inside the app container once it’s up:

```bash
docker compose run --rm app python3 scripts/create_gmail_auth_config.py
```

**Option 2: Composio dashboard (manual)**

If you’d rather click through the UI:

1. Dashboard → **Auth Configs** → **New Auth Config** → **Gmail**.
2. **OAuth App type**: **Use Composio-managed OAuth app**.
3. Keep the default Gmail scopes (read/search messages, profile, contacts).
4. Save, copy the **Auth Config ID**, paste into `.env`.

> No callback URL needs to be registered anywhere on the Composio side — the app passes `callback_url` dynamically at runtime based on `BASE_URL`.

### 5c. Make sure your Gmail account can sign in

Composio-managed OAuth apps are published by Composio, so in most cases any Gmail account can sign in. If you hit a Google “access blocked / app not verified” screen:

- Try a different Gmail account (your personal one usually works).
- If it persists, contact Composio support — only they control that OAuth app.

---

## 6. Start an HTTPS tunnel with ngrok

Google OAuth requires HTTPS for the redirect URL. ngrok gives you a free public HTTPS URL that forwards to your local app.

In a **separate terminal**:

```bash
ngrok http 8080
```

You’ll see output like:

```
Forwarding    https://abcd-1234.ngrok-free.app -> http://localhost:8080
```

Copy that HTTPS URL (without a trailing slash) and put it in `.env` as `BASE_URL`:

```
BASE_URL=https://abcd-1234.ngrok-free.app
```

> Keep the ngrok terminal running while you use the app. Every time you restart ngrok the URL changes — update `BASE_URL` and restart the app when that happens.

---

## 7. Run the app

From the project root:

```bash
docker compose up
```

First run will download images and install Python deps — give it a minute or two. When ready, you’ll see logs from both `mongo` and `app`, including:

```
Uvicorn running on http://0.0.0.0:8000
```

(Internally it runs on port 8000; Docker maps it to **8080** on your host.)

Open either:

- `http://localhost:8080` for quick health checks, **or**
- your ngrok HTTPS URL for the full flow (required for Google sign-in).

### Try the flow

1. Go to your ngrok URL in the browser.
2. Click **Connect Gmail** → complete Google sign-in.
3. You’re redirected back into the app; enrichment runs in the background.
4. Watch the Docker logs — you should see:
   - `Started background enrichment for entity: ...`
   - `Successfully completed enrichment for entity: ...`
5. Start chatting. The first reply should reference something specific from your mailbox once the status flips to `active`.

To stop: `Ctrl+C` in the compose terminal, then `docker compose down` (add `-v` to also wipe the Mongo volume).

---

## 8. Troubleshooting

**`redirect_uri_mismatch` or Google blocks sign-in**
`BASE_URL` probably doesn’t match the URL you actually opened in the browser. Make sure you’re visiting the **same** HTTPS URL that’s in `.env`, and restart the app after changing it.

**Stuck on `enriching` forever**
Check the app logs. Most often: missing/invalid `OPENAI_API_KEY` or `PERPLEXITY_API_KEY`, or no credits on those accounts. The app is designed to fail silently for the user — logs are where you’ll see the real error.

**Mongo connection errors on startup**
Make sure you ran `docker compose up` (not just `docker run` on the app image). The compose file wires `mongo` and `app` together on the same network.

**ngrok URL expired / changed**
Update `BASE_URL` in `.env` and run `docker compose up` again (compose picks up env changes on restart).

**Changed `.env` but the app doesn’t see it**
Environment variables are read at container start. `Ctrl+C` the running compose stack and `docker compose up` again.

---

## 9. What each key is actually used for

| Variable | Purpose |
|---|---|
| `COMPOSIO_API_KEY` | Server-side auth with Composio (creating auth links, fetching Gmail data). |
| `COMPOSIO_GMAIL_AUTH_CONFIG_ID` | Points at the Gmail integration you configured in the Composio dashboard. |
| `OPENAI_API_KEY` | Extraction (per-email-bucket structured JSON), persona writing, chat replies. |
| `PERPLEXITY_API_KEY` | Public web profile lookup (role, company, LinkedIn, bio). |
| `MONGODB_URL` | Where user profiles + chat history are stored. Compose overrides this to the internal container. |
| `BASE_URL` | Public origin used to build the Composio OAuth callback URL. Must be HTTPS for Google. |

---

## 10. Minimum viable checklist

- [ ] Docker Desktop running
- [ ] `.env` populated with **all 6** values
- [ ] ngrok tunnel up, `BASE_URL` matches the ngrok HTTPS URL
- [ ] OpenAI and Perplexity accounts have credit
- [ ] Composio Gmail auth config created and its ID pasted into `.env`
- [ ] `docker compose up` running with no errors in the logs

If all six are green, you should be able to sign in and chat.
