# Universal OAuth 2.0 — Design

## Status

Implemented. 654 unit tests, plus a scripted end-to-end run against a fake provider
covering mint → consent → callback → authenticated call → disconnect.

**Not yet run against a real provider.** Strava is the first target — see
[Testing](#testing).

The agent-facing skill ships as runtime data in `/app/data/skills/`, not in this
repo.

The existing Gmail flow (`aug/api/routers/gmail_auth.py`) is deliberately left in
place and is not migrated — see [Out of scope](#out-of-scope).

---

## Problem

AUG should be able to talk to any OAuth 2.0 API — Spotify, Strava, Google Calendar,
whatever comes next. Each needs the same four things: a callback endpoint, a token
exchange, encrypted storage, unattended refresh.

The one existing implementation was written for Gmail and is not reusable: `state`
carries an account nickname rather than a CSRF token, there is no PKCE, and tokens
sit in plaintext on disk. Building a second bespoke flow per provider does not
scale. This specifies one flow for all of them.

## Goals

1. **One implementation, any provider.** Adding a provider is config plus two
   secrets — no Python.
2. **Safe on a public API.** The callback is necessarily unauthenticated; it must be
   inert without a valid, single-use, server-side `state`.
3. **Correct refresh.** Some providers rotate the refresh token on every use, where
   a lost update permanently disconnects the account.
4. **The agent uses credentials without reading them.** Not because the agent is
   assumed hostile, but because a destination-bound capability beats a loose bearer
   string and here it costs almost nothing.

## Non-goals

- **Defending against host compromise.** AUG restarts unattended, so it can always
  reach its own key, so root can too. Encryption at rest raises the bar against
  stolen backups and nothing more.
- **Preventing an injected agent from exfiltrating data it legitimately read.**
  Response bodies flow into the model; that is inherent to having tools.
- **Being a general egress control.** The gateway is not a forward proxy.

---

## Standards baseline

| Requirement | Source | How it is met |
|---|---|---|
| PKCE on all clients | RFC 9700 | S256, verifier server-side in the state row |
| Exact redirect URI matching | RFC 9700 | One URI per provider, built from `BASE_URL` |
| Authorization code only | OAuth 2.1 | No implicit, no password grant |
| Mix-up defence | RFC 9700 | Provider in the path, matched against the state row; `iss` checked when sent |
| `iss` response parameter | RFC 9207 | Validated when present; absence not fatal (most consumer providers omit it) |
| CSRF protection | RFC 9700 | 256-bit `state`, server-side, single-use, 10 min TTL |

---

## Architecture

Six modules, all in the existing AUG process:

```
aug/utils/oauth.py            protocol client — authorize URL, exchange, refresh, revoke
aug/utils/ratelimit.py        token-bucket limiter for the public endpoints
aug/core/oauth/providers.py   provider registry — the only writer of the config file
aug/core/oauth/store.py       token store — encrypted Postgres persistence
aug/core/oauth/refresh.py     on-demand refresh, one lock per (provider, account)
aug/api/internal/gateway.py   loopback gateway — 127.0.0.1:8799, all agent-facing routes
aug/api/routers/oauth.py      public flow — /start and /callback only
```

Per the layering rule in CLAUDE.md: how to talk to a provider lives in `utils/`,
what to do with the result in `core/`, HTTP surface in `api/`.

**Why `api/internal/` and not `api/routers/`.** Everything in `routers/` is mounted
publicly by `create_app()`. A credential-attaching endpoint there would be
internet-reachable, defended only by a reverse-proxy rule — exactly what the loopback
bind exists to make impossible. A sibling `internal/` package states that in the
directory layout rather than in a comment. The cost, accepted: the subsystem spans
two trees.

### Why a loopback gateway rather than environment variables

Once a token is `$SPOTIFY_ACCESS_TOKEN`, its authority is unbounded: any destination,
any process in the container, any `set -x`, anything reading `/proc/*/environ`.
Behind `localhost:8799/spotify/`, its authority is exactly "one pinned host, one
credential, refreshed correctly."

A forward proxy with TLS interception, as OpenClaw does, achieves the same binding
but requires minting a CA and rewriting container-wide certificate trust — a bug
there breaks *every* HTTPS call in the container. Its own authors ship it disabled by
default. The path-prefix gateway gets the same property for ~80 lines with a blast
radius of "OAuth calls stop working."

---

## Data model

Three tables, created in `_ensure_schema()`.

```sql
CREATE TABLE IF NOT EXISTS oauth_start_tokens (
    token       TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    account     TEXT NOT NULL DEFAULT 'primary',
    expires_at  TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS oauth_states (
    state         TEXT PRIMARY KEY,
    provider      TEXT NOT NULL,
    account       TEXT NOT NULL DEFAULT 'primary',
    code_verifier TEXT NOT NULL,
    redirect_uri  TEXT NOT NULL,
    issuer        TEXT,
    expires_at    TIMESTAMPTZ NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS oauth_tokens (
    provider          TEXT NOT NULL,
    account           TEXT NOT NULL DEFAULT 'primary',
    access_token_enc  BYTEA NOT NULL,
    refresh_token_enc BYTEA,
    token_type        TEXT NOT NULL DEFAULT 'Bearer',
    scopes            TEXT NOT NULL DEFAULT '',
    issuer            TEXT,
    expires_at        TIMESTAMPTZ,
    needs_reauth      BOOLEAN NOT NULL DEFAULT FALSE,
    last_error        TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (provider, account)
);
```

Plus expiry indexes on both short-lived tables.

### Encryption

AES-256-GCM. Key from `OAUTH_ENCRYPTION_KEY` (32 bytes, base64), a plain Docker env
var — **not** in `/app/data`, so a stolen volume snapshot is useless alone. Nonce is
12 random bytes, prepended. **AAD is `f"{provider}|{account}"`**, so a row copied over
another fails to decrypt rather than silently authenticating as the wrong account.

Honest scope: this protects backups and DB-side compromise. It does not protect
against the agent, which can read `$OAUTH_ENCRYPTION_KEY` from its own environment.
Accepted.

---

## The flow

### 1. Minting a start link

`/start` cannot require `X-API-Key` — it opens in a browser — and leaving it open is
what makes a flow graftable. Minting therefore happens on the gateway, which is
authenticated by being loopback-only:

```
POST localhost:8799/_link/spotify  →  {"url": ".../oauth/spotify/start?t=..."}
```

32 random bytes, 10 min TTL, single use.

**No Telegram command and no public `/link`.** An earlier draft had both. A slash
command can mint a link but cannot tell you where to register the app, that Strava
separates scopes with commas, or that Google needs `access_type=offline` — all of
which the agent can research. The flow belongs in conversation with the agent; a
second entry point doing half of it is a hardcoded value with extra steps.

An injected agent can mint links, but a link is inert until the user completes
consent with their own account at the provider.

### 2. `GET /oauth/{provider}/start?t=<token>`

1. Claim the start token (`DELETE … RETURNING`). No row → `400`.
2. Look up the provider. Unknown → `400`.
3. Generate `state` (32 bytes) and a PKCE verifier; insert into `oauth_states` with
   provider, account, exact `redirect_uri` and expected `issuer`, 10 min TTL.
4. `302` to the authorize URL with `response_type=code`, `client_id`, `redirect_uri`,
   `scope`, `state`, `code_challenge`, `code_challenge_method=S256`, plus any
   `extra_authorize_params`.

### 3. `GET /oauth/{provider}/callback`

Unauthenticated by necessity. Ordering matters: **every cheap rejection happens
before any outbound call**, so an unknown `state` costs one indexed DELETE.

1. Rate limit by client IP.
2. Claim the state atomically: `DELETE FROM oauth_states WHERE state = $1 AND
   expires_at > NOW() RETURNING *`, requiring exactly one row. Single-use and
   race-free.
3. Assert the row's `provider` equals the path segment → mix-up defence.
4. If the provider sent `iss`, assert it matches the row's `issuer`.
5. **Only now** POST to the token endpoint with `grant_type=authorization_code`,
   `code`, `redirect_uri`, `code_verifier` and client credentials. Redirects are not
   followed.
6. Upsert into `oauth_tokens`, encrypting both tokens and recording the **granted**
   scopes — providers may grant less than requested.
7. Render a success page.

A user who declines consent is sent back with `error` and **no** `code`, so `code`
is optional: requiring it would answer a deliberate choice with a validation error
page. The error branch claims the state, logs, and renders a plain "not connected"
page without any token exchange.

Every callback response carries `Referrer-Policy: no-referrer`, `Cache-Control:
no-store`, `Content-Security-Policy: default-src 'none'`, and loads **zero external
assets** — no fonts, no favicon — because the `code` is in the URL and would leak via
`Referer` to anything fetched. `code` and `state` are scrubbed from logs.

### Rate limiting

`aug/utils/ratelimit.py` — in-process token bucket, no new dependency. 10/min/IP on
the callback, 30/hour/IP on start, checked **before the state lookup** so a flood
buys no database work.

Client IP is the **rightmost** `X-Forwarded-For` entry. With one trusted proxy hop
everything to its left is caller-forgeable; taking the leftmost would let an attacker
mint a fresh allowance per request by prepending a random address.

Two properties that are easy to get wrong, both tested:

- **The store is bounded** (10k entries). An unbounded dict keyed by client IP makes
  the rate limiter the memory-exhaustion vector it was added to prevent.
- **Eviction is by fullness, not age.** A refilled bucket is indistinguishable from
  an untracked caller, so it goes first. LRU would evict the caller being actively
  limited — usually the oldest entry — resetting an attacker's allowance.

Rate limiting is the *second* line of defence. The first is step 2 above.

### `BASE_URL` enforcement

`config.py` rejects a non-HTTPS `BASE_URL` when `DEBUG` is false, failing the boot.
`BASE_URL` defaults to `""`, so without this a misconfigured deploy mints hostless
links and nobody finds out until someone taps one. Providers refuse to register a
non-HTTPS redirect URI anyway, so the failure would only surface later and more
confusingly. `docker-compose.prod.yml` requires the variable explicitly rather than
defaulting it.

---

## Token gateway

A second `uvicorn.Server` on `127.0.0.1:8799`, same process and event loop, started
as a task in `lifespan`.

**Always listening.** An earlier draft started it lazily, once a token existed. That
was reversed: with nothing connected the gateway can answer only "nothing connected"
and `404`, so there is nothing behind the socket to protect — while a refused
connection is indistinguishable to the agent from a crashed gateway, and it would
report a broken gateway as a dead provider. Listening unconditionally also removes
the need for a second status endpoint on the public API.

**Failure is non-fatal.** The socket is bound explicitly before uvicorn receives it,
because uvicorn calls `sys.exit` when it cannot bind and `SystemExit` escapes
`except Exception` — a port conflict would otherwise kill AUG at boot. It stops
before the asyncpg pool closes.

**Why loopback and not a path on the main app.** Port 8000 sits behind the public
proxy; a credential-attaching endpoint there would be internet-reachable, one typo
from exposure. `run_bash`'s shell is a child process in the same container, so
`localhost` resolves correctly; other compose services cannot reach it.

### Routes

```
GET    /                       → connected providers and accounts, plus what is configured
ANY    /{provider}/{path...}   → authenticated proxy to the provider's api_base
DELETE /{provider}             → revoke + delete
POST   /_link/{provider}       → mint a single-use start link
POST   /_provider/{provider}   → validate and save one provider's config
DELETE /_provider/{provider}   → forget a provider's config
```

Account selection via `X-Aug-Account`, defaulting to `primary`, on every route.

Two shapes differ from the first draft, both to remove a collision:

- **`/_link/`, `/_provider/`** rather than `/{provider}/link` — a provider whose own
  API has a `/link` endpoint would otherwise be silently shadowed.
- **`DELETE /{provider}`** rather than `DELETE /{provider}/{account}` — the latter is
  ambiguous with a proxied delete (`DELETE /spotify/v1/tracks/123` must reach
  Spotify). The header already selects the account everywhere else.

### Request handling

1. Resolve the provider; unknown → `404` naming the ones that exist.
2. Load the token, refreshing if within 60s of expiry. A token that cannot be
   decrypted — `OAUTH_ENCRYPTION_KEY` changed — returns `503` naming that cause,
   rather than a stack trace the agent would read as a provider outage.
3. Build the upstream URL against `api_base`. **Reject anything that would change the
   host** — absolute URLs, `..`, and protocol-relative paths.
4. Forward method, query, body and a safe subset of headers, adding `Authorization`.
   Strip client-supplied `Authorization`, `Host`, `Cookie`.
5. **Do not follow redirects.** A 3xx returns as-is; following it would carry the
   auth header off the pinned host.
6. Timeouts: 10s connect, 30s total.

### Scope

Available to `run_bash` only, deliberately. `run_ssh` runs on a remote machine whose
`localhost` is its own, and stock `sshd` drops non-`AcceptEnv` variables — so no
mechanism reaches it. That is the desired outcome: refresh tokens should not be
copied onto remote hosts with their own users and backups.

---

## Refresh

**On demand at point of use, never a background loop.** A loop refreshes credentials
nobody is using, multiplying exposure to rotation failures for no benefit, and still
cannot save you from a token the provider killed early.

```
async with lock_for(provider, account):     # asyncio.Lock per (provider, account)
    row = await load(provider, account)     # re-read INSIDE the lock
    if not expiring_within(row, 60s):
        return row                          # someone else already refreshed
    return await exchange_and_save(row)
```

In-process locks suffice because AUG is a single process; if that changes, this
becomes a Postgres advisory lock.

### Rotating refresh tokens

Some providers (Strava) issue a new refresh token on every refresh and invalidate the
old one. Two consequences:

- **The store must be transactional.** This is why tokens live in Postgres rather
  than hushed: hushed's `Load → Add → Save` rewrites the whole encrypted file with no
  locking, and a lost update on a rotating refresh token does not mean a stale value
  — it means the account is **permanently disconnected**.
- **Exactly one owner.** If the same client is authorized in AUG *and* elsewhere they
  will fight, and one will be randomly logged out.

### Failure handling

- **Reuse race** — error contains `reused` or "already been used". Retry once; do not
  mark the account dead.
- **Real revocation** — `invalid_grant` and friends. Set `needs_reauth = TRUE`, store
  `last_error`. **Do not delete the row**: it is what lets the gateway return an
  actionable error.

The gateway then answers `503 spotify/primary requires re-authorization (refresh
failed: invalid_grant)`, satisfying the tool standard in CLAUDE.md — never something
that reads as success when the task did not complete.

---

## Disconnecting

Deleting the local row is **not** revocation: the grant stays live and the refresh
token remains valid indefinitely. A real disconnect is revoke-then-delete.

`DELETE localhost:8799/{provider}`, with `X-Aug-Account` selecting the account. A REST
endpoint can be added when the Mini App needs one.

No approval gate: worst case from an injected agent is a disconnect you undo in a
minute, and "disconnect strava" should just work.

Revocation is not universal — Google has an endpoint, Strava has `deauthorize`,
Spotify has none. `revoke_url` is optional, and when absent the response says so
plainly rather than implying a full disconnect:

```
Deleted spotify/primary from AUG. The app's access was NOT revoked —
remove it in your spotify account settings.
```

If revocation fails, delete locally anyway — otherwise a dead row is unremovable —
and return the same message.

---

## Provider configuration

`/app/data/oauth_providers.json`, owned by `ProviderRegistry`. The agent never writes
it directly; it posts to `POST localhost:8799/_provider/{name}`, which validates
through the `ProviderConfig` model before anything reaches disk, merges rather than
replaces, and writes atomically. A rejected config returns field-level errors:

```
400 Invalid provider config:
token_url: must be https:// — got 'http://evil.example/token'
```

**Read on every lookup, not cached.** The file is ~1KB and page-cached; a copy would
buy nothing and cost an invalidation rule. It also means a provider added
mid-conversation is usable in the next step, with no restart — which matters, because
that restart would land in the middle of the connect flow.

OAuth 2.0 has no universal discovery, and the quirks are real: Spotify separates
scopes with spaces, Strava with commas; Google returns no refresh token without
`access_type=offline&prompt=consent`. Registering the app by hand is irreducible
anyway, so supplying a few fields at the same moment is nearly free.

```json
{
  "strava": {
    "authorize_url":   "https://www.strava.com/oauth/authorize",
    "token_url":       "https://www.strava.com/oauth/token",
    "api_base":        "https://www.strava.com/api",
    "revoke_url":      "https://www.strava.com/oauth/deauthorize",
    "scopes":          ["read", "activity:read_all"],
    "scope_separator": ","
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `authorize_url`, `token_url`, `api_base` | yes | HTTPS only |
| `scopes` | yes | Provider-specific strings |
| `issuer` | no | Expected `iss` value |
| `revoke_url` | no | Absent → local delete with an honest message |
| `scope_separator` | no | Defaults to `" "` |
| `extra_authorize_params` | no | Provider quirks |
| `token_auth_method` | no | `body` (default) or `basic` |
| `client_id_env`, `client_secret_env` | no | Default `{PROVIDER}_CLIENT_ID` / `_SECRET` |

Client credentials live in hushed, never in this file — it is plaintext in an
agent-readable directory.

**Validation.** Every URL must be HTTPS. The registry still skips a malformed entry
with a loud log at read time, because the file can be hand-edited and one bad
provider must not disable every other integration.

**Reconfiguring a connected provider is refused** with `409`; disconnect first.
Rewriting `token_url` on a live provider would redirect its refresh token at the next
renewal. An earlier draft accepted that risk on the grounds that hushed already
injects `{PROVIDER}_CLIENT_SECRET` into every `run_bash` call — true for the secret,
but not for the refresh token, which exists only in the database.

**This is not enforcement.** `run_bash` can still write the file directly — same
user, same container, and the blocklist is a substring match. What the endpoint buys
is that the correct path is easy and gives good errors, so nobody has a reason to
hand-edit. Real enforcement would mean running `run_bash` as a different user.

---

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OAUTH_ENCRYPTION_KEY` | If OAuth is used | 32 bytes base64. Plain Docker env var, **not** in `/app/data`. Losing it means re-authorizing every provider. |
| `BASE_URL` | Yes | Must be `https://` in production. |
| `{PROVIDER}_CLIENT_ID` / `_SECRET` | Per provider | In hushed. |

---

## Out of scope

- **The Gmail flow is not migrated.** `gmail_auth.py`, `gmail_credentials.py` and the
  typed Gmail tools stay as they are, by explicit decision. Two OAuth implementations
  therefore coexist, and the older one does not meet the standards baseline above.
  Migrating it is the obvious follow-up.
- **`run_bash` blocklist hardening** — a separate few-line change. It stops accidental
  disclosure, not deliberate reading.
- **A `SKILL.md` in this repo.** Skills live in `/app/data/skills/` as runtime data.
- **Device Authorization Grant** (RFC 8628). Sidesteps callbacks entirely; worth
  adding later, but none of the initial targets support it.
- **Multi-user.** AUG has no `user_id`; the key is `(provider, account)`.

---

## Testing

**Unit — 646 tests**, mocked at the HTTP boundary with `httpx.MockTransport` so the
real request-building code runs (form encoding, Basic-vs-body auth,
`follow_redirects=False`) rather than being asserted through mock call args. This is a
new pattern here; the rest of the suite patches `httpx.AsyncClient` directly.

Covered: single-use state, expired state rejected with no outbound call, mix-up
defence, `iss` mismatch and absence, AES-GCM round trip and AAD binding, host pinning
against traversal and protocol-relative paths, redirects returned not followed,
client `Authorization` stripped, `needs_reauth` → `503`, concurrent refresh collapsing
to one exchange, rotation persisted, `invalid_grant` flagged and retained, reuse race
retried once, config validation and the `409` guard, rate-limit exhaustion and
bounded eviction.

**Deliberately not covered: the SQL.** There are no Postgres integration tests,
matching the rest of the suite — CI has no database. So the `DELETE … RETURNING`
atomicity that the single-use `state` guarantee rests on is unverified, as is whether
any query matches the schema. This is the one place that gap touches security-critical
code.

**Four defects the tests and smoke runs caught** that review had not: a
protocol-relative path escaping the pinned host; `expires_in: 0` read as "no expiry"
rather than "already expired"; uvicorn's `sys.exit` on bind failure killing the boot;
and an undecryptable token surfacing as an opaque `500`.

**Scripted end-to-end against a fake provider** (`httpx.MockTransport` standing in for
Spotify, with an in-memory row store so data survives between requests): mint → start
→ consent redirect → callback → token stored encrypted → authenticated API call with
the token attached → status → replayed code refused → disconnect. This is what proved
the pieces fit together; the unit tests each mock one seam.

**End-to-end — not yet run.** Strava first: free, simple, and it rotates refresh
tokens, so it exercises the hardest path.

1. `POST /_link/strava` → open → authorize → confirm the row lands encrypted.
2. `curl localhost:8799/strava/api/v3/athlete` from `run_bash` returns real data.
3. Force expiry in the DB, call again, confirm exactly one refresh and a new refresh
   token persisted.
4. Revoke at Strava's site, call again, confirm a `503` naming `invalid_grant` — not a
   false success.
5. `DELETE localhost:8799/strava` → confirm deauthorized at the provider.

---

## Open questions

- Should something surface `needs_reauth` accounts proactively, or only when the agent
  hits one?
- Is a periodic sweep of expired `oauth_states` / `oauth_start_tokens` worth a
  scheduler entry? Nothing currently deletes rows that are minted and never used.
- Should the gateway log every authenticated call as an audit trail — structured logs,
  or a table?
- Is the `409` guard on reconfiguring a connected provider worth its friction, or does
  it just push the agent toward hand-editing the file it cannot be stopped from
  writing?
- Should the rate limiter be replaced by the `limits` library, or does its bounded,
  fullness-evicting store justify keeping ~45 lines in-house?
