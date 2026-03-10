# AUG Tools

| Tool | Status | Description |
|------|--------|-------------|
| `brave_search` | ✅ Done | Web search via Brave Search API. Returns top N results with title, URL, and snippet. Requires `BRAVE_API_KEY`. |
| `fetch_page` | ✅ Done | Fetches and extracts readable content from one or more URLs simultaneously. Uses trafilatura for content extraction. Includes links in output. |
| `get_current_datetime` | ✅ Done | Returns current UTC datetime. Kept for explicit time queries; time is also injected into every message via `TimeAwareChatAgent`. |
| `run_python` | 💡 Planned | Execute arbitrary Python in a sandbox. Solves math, data analysis, file manipulation — anything computable. |
| `run_bash` | 💡 Planned | Full shell access inside the Docker container. Most powerful single tool. Should redact secret env var values from output. Requires a fuller base image (curl, jq, git, etc.). |
| `http_request` | 💡 Planned | Generic GET/POST to any URL/API. Massively unlocks integrations without needing a dedicated tool per service. Agent can use env vars for auth headers. |
| `browser` | 💡 Planned | Headless browser (Playwright) for JS-heavy sites, SPAs, login flows, and button clicks. Complements `fetch_page` for the ~20% of sites that require JS rendering. Adds ~500MB (Chromium) to the image. |
| `remember` | ✅ Done | Save a memory with a short description and full content. Stored in `data/memories.json`. Index injected into every system prompt. |
| `recall` | ✅ Done | Retrieve the full content of a memory by its id. |
| `forget` | ✅ Done | Delete a memory permanently by its id. |
| `read_file` | 💡 Planned | Read a file from the local filesystem inside the container. |
| `write_file` | 💡 Planned | Write or modify a file on the local filesystem inside the container. |
