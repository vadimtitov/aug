# Gmail Setup

Gives the agent access to your Gmail — search, read, send, and draft emails.

---

## 1. Create a Google Cloud project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown (top left) → **New Project** → name it `AUG` → **Create**
3. Go to **APIs & Services → Library** → search **Gmail API** → **Enable**

---

## 2. Configure OAuth consent screen

1. Go to **APIs & Services → OAuth consent screen**
2. Choose **External** → **Create**
3. Fill in:
   - App name: `AUG`
   - User support email: your Gmail
   - Developer contact email: your Gmail
4. Click **Save and Continue**
5. Click **Add or Remove Scopes** → paste `https://mail.google.com/` → **Add → Save and Continue**
6. Under **Test users** → **Add users** → add every Gmail address you want to use → **Save and Continue**

---

## 3. Create OAuth credentials

1. Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**
2. Application type: **Web application**
3. Name: `AUG`
4. Under **Authorized redirect URIs** → add your callback URL:
   - Local dev: `http://localhost:8012/auth/gmail/callback`
   - Home server: `http://YOUR_SERVER_IP:8012/auth/gmail/callback`
5. Click **Create** → copy the **Client ID** and **Client Secret**

---

## 4. Add to your environment

Add to your `.env` (or Portainer environment variables):

```
GMAIL_CLIENT_ID=your-client-id
GMAIL_CLIENT_SECRET=your-client-secret
BASE_URL=http://YOUR_SERVER_IP:8012
```

`BASE_URL` is the address where your AUG server is reachable from a browser. Used to build the OAuth callback and auth links.

---

## 5. Connect your Gmail account

Start AUG, then open this URL in a browser:

```
http://YOUR_SERVER_IP:8012/auth/gmail
```

- Google will show a consent screen (may say "app not verified" — that's fine, click **Continue**)
- Pick your Google account and grant access
- You'll see: `{"status": "ok", "account": "primary"}`

Done. The token is saved and auto-refreshes — you won't need to do this again.

### Multiple accounts

To connect a second account, visit:

```
http://YOUR_SERVER_IP:8012/auth/gmail?account=work
```

Then refer to it in prompts: *"check my work email for invoices"*.

---

## 6. Use it

Just ask:

> "Do I have any unread emails from this week?"
> "Draft a reply to the last email from John saying I'll be there"
> "Search my work Gmail for invoices from March"

If Gmail isn't connected yet, the agent will reply with a link to authorize.
