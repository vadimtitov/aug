# Browser tool

AUG can control a real Chrome browser to complete tasks: log into websites, fill forms, add items to baskets, scrape JS-rendered pages, and anything else you'd normally do manually.

The browser runs in a Docker container alongside AUG. You get a web UI to watch it work in real time and log in to sites manually when needed.


## How it works

- Chrome runs inside the `chromium` container, controlled via the Chrome DevTools Protocol (CDP)
- AUG's browser tool sends tasks to it in plain language ("add Diet Coke to my Amazon basket")
- A sub-agent (browser-use) drives Chrome step by step to complete the task
- While it works, Telegram shows live step updates so you know what it's doing
- You can watch — or take over — at any time via the browser UI on port 3012


## First-time setup (production)

**1. Create the Chrome profile directory on your server:**

```bash
sudo mkdir -p /opt/aug/chrome-profile && sudo chown -R 1000:1000 /opt/aug/chrome-profile
```

This is where Chrome stores cookies, sessions, and logins. Without it, you'd have to log in again every time the container restarts.

**2. Set environment variables in Portainer:**

| Variable | Required | Description |
|----------|----------|-------------|
| `BROWSER_CDP_URL` | Yes | `http://chromium:9222` |
| `CHROME_PASSWORD` | Yes | Password for the browser UI on port 3012 |
| `CHROME_USER` | No | Username for the browser UI (default: `aug`) |

**3. Deploy the stack.** Portainer will build and start both `aug` and `chromium`.

**4. Open the browser UI** at `http://your-server:3012` — log in with your `CHROME_USER` / `CHROME_PASSWORD`. You'll see a live Chrome window running on your server.

**5. Log into any websites you want AUG to access.** Google, Amazon, whatever. Do it here exactly as you would in a normal browser, including any 2FA prompts on your phone. Sessions are saved to the profile directory and persist across restarts.

That's it. AUG can now use those sessions whenever you ask it to do something on those sites.


## Logging into sites with 2FA

For Google and other accounts that use push-notification 2FA ("tap to confirm on your phone"), there's no way to automate that step — it requires you physically approving it.

The practical approach:
1. Open port 3012
2. Log in manually, approve 2FA on your phone
3. Close the tab — session is saved
4. AUG uses the session from now on without needing credentials or 2FA

Google sessions typically last weeks to months. When one expires, just repeat.

For sites that use **TOTP codes** (Google Authenticator-style 6-digit codes), AUG can handle that automatically — see [Credentials and secrets](#credentials-and-secrets) below.


## Credentials and secrets

AUG never receives your passwords directly. Credentials are stored in [hushed](https://github.com/vadimtitov/hushed) (an encrypted local secret store) and injected into the browser at runtime — the LLM only ever sees placeholder names like `{email}`, never the actual values.

**Storing a secret via Telegram:**

```
/secret
```

Follow the prompts. The name you give it (e.g. `AMAZON_EMAIL`) is the env var name you'll reference in tasks.

**Using secrets in a task:**

```
Add Diet Coke 24-pack to my Amazon basket.
Log in with email {email} and password {password}.
Use secrets: email=AMAZON_EMAIL, password=AMAZON_PASSWORD
```

AUG translates this into a `browser` tool call with the right placeholders and secret mappings. The actual values are substituted inside the browser tool without ever passing through the LLM.

**TOTP / authenticator codes:**

If a site uses a TOTP authenticator app, store the TOTP seed secret in hushed with a name ending in `_bu_2fa_code`:

```
/secret
Name: AMAZON_2FA_BU_2FA_CODE
Value: <your TOTP seed>
```

Then reference `{amazon_2fa}` with `amazon_2fa=AMAZON_2FA_BU_2FA_CODE` in the task. AUG generates a live 6-digit code automatically at login time.


## Watching AUG work

When AUG is using the browser, Telegram shows a live status message that updates each step:

```
🕐 Browser(add Diet Coke to basket)
Step 4 · amazon.co.uk
Clicking "Add to Basket" button
```

You can also open port 3012 at any time to watch Chrome directly. If AUG is mid-task, you'll see it working in real time.


## Security notes

- Port 3012 (browser UI) is protected with HTTP basic auth. Set a strong `CHROME_PASSWORD`.
- Port 3012 uses HTTP, not HTTPS. If accessed only on your local network or VPN, this is fine. If exposed directly to the internet, put it behind a reverse proxy with TLS (Nginx/Traefik/Caddy).
- Port 9222 (CDP) is internal — not exposed to the host in production. Only `aug` and `chromium` can reach it.
- Chrome runs with `--no-sandbox`. This is standard practice for containerised Chrome and acceptable in a trusted environment.


## Changing the browser model

The browser sub-agent uses `gemini-2.5-flash` by default. To change it, set via the AUG settings API or Telegram tooling:

```json
{ "tools": { "browser": { "model": "gpt-4.1" } } }
```

`gpt-4.1` and `gpt-4o` are the most reliable choices for complex multi-step tasks.
