# Skills Management UI — Design Spec

Mini app UI for browsing, viewing, editing, and installing skills from ClawHub.

---

## Navigation

- **Entry point:** Replace the "Models" tile on the HomePage with "Skills" (keeps 2×2 grid).
- **Routing:** `App.tsx` manages a `pages: PageState[]` stack with `navigate(page, params)` / `goBack()`. No router library.

---

## Page Structure & Flow

```
Home
└── SkillsPage (tabs: Mine | ClawHub)
    ├── [Mine tab] Local skill list
    │   └── SkillDetailPage (local)
    │       ├── In-place edit mode (description field, always-on toggle, body textarea)
    │       ├── Delete (confirmation dialog)
    │       └── FileViewerPage
    │           └── In-place edit mode (textarea) + Delete (confirmation dialog)
    └── [ClawHub tab] Trending list + search bar
        └── SkillDetailPage (clawhub)
            ├── Owner, version, moderation badge, stats
            ├── SKILL.md rendered (read-only)
            ├── File list (read-only, viewable)
            └── Install / Update button (warns if already installed with local edits)
```

---

## Skill Detail Page — Local Skill

Layout (top to bottom):

1. **Name** — read-only header
2. **Description** — editable text field, saves on blur
3. **Always-on toggle** — labeled "Always active — inject into every prompt", saves immediately
4. **SKILL.md body** — rendered markdown in view mode; switches to textarea in edit mode. Pencil icon in top-right toggles edit mode. Save / Cancel appear when editing.
5. **Supporting files list** — each file tappable → FileViewerPage
6. **Delete button** — bottom, danger style, triggers confirmation dialog

Edit mode rules:
- Textarea contains **body only** (no YAML frontmatter — that's handled by the structured fields above).
- Description and always-on are always editable regardless of edit mode.
- Backend reassembles the full SKILL.md from structured fields + body on save.
- Name is **read-only** (it's the directory name and skill identity).

---

## Skill Detail Page — ClawHub Skill

Layout (top to bottom):

1. **Name** + owner handle + version
2. **Moderation badge** — "Clean" or "Suspicious" (from `moderation.verdict`)
3. **Download count** + stars
4. **Description**
5. **SKILL.md body** — rendered markdown, read-only
6. **Supporting files list** — read-only, each tappable → FileViewerPage (read-only)
7. **Install / Update / Installed button**
   - Not installed → "Install"
   - Installed, same version → "Installed" (disabled)
   - Installed, newer version on ClawHub → "Update" → warns "This will overwrite your local copy. Continue?"
   - Overwrite warning shown if local edits detected (file modified time vs install time)

---

## File Viewer Page

- Monospace `<pre>` block for content display (view mode)
- Pencil icon top-right → edit mode (textarea, full file content)
- Save / Cancel in edit mode
- Syntax highlighting via `react-syntax-highlighter` light build (Python, bash, JS, Markdown)
- Delete button at bottom with confirmation dialog
- Supported operations: **view, edit content, delete**. No rename, no add new file.

---

## ClawHub Tab

- **On load:** Trending skills (`GET /api/v1/skills?sort=trending&limit=20`)
- **Search bar at top:** Typing switches to semantic search (`GET /api/v1/search?q=...`), clears back to trending on empty
- **"Load more" button** at list bottom for pagination (cursor-based)
- **Skill cards** show: name, description summary, version, download count
- **"Installed" badge** on cards for skills already present locally
- All ClawHub read calls (`clawhub.ai/api/v1/*`) made **directly from the browser** — CORS is enabled, no auth required for reads
- Only the install action goes through the AUG backend

---

## Libraries Added

| Library | Purpose |
|---------|---------|
| `react-markdown` | Render SKILL.md body safely |
| `remark-gfm` | GitHub-flavored markdown (tables, task lists, strikethrough) |
| `react-syntax-highlighter` (light build) | Code highlighting in FileViewerPage |

---

## Backend — New Endpoints (`aug/api/routers/skills.py`)

All endpoints require JWT auth (same as existing routes).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/skills` | List local skills (name, description, always_on, file count) |
| `GET` | `/skills/{name}` | Skill detail + file list |
| `GET` | `/skills/{name}/file?path=` | Raw file content |
| `PUT` | `/skills/{name}` | Update description, always_on, and/or body (backend reassembles SKILL.md) |
| `PUT` | `/skills/{name}/file?path=` | Update a supporting file's content |
| `DELETE` | `/skills/{name}` | Delete entire skill directory (with confirmation on frontend) |
| `DELETE` | `/skills/{name}/file?path=` | Delete a supporting file |
| `POST` | `/skills/{name}/install` | Download ClawHub zip + extract to disk |

The `POST /skills/{name}/install` body: `{ slug: string, version?: string }`. Backend calls `GET https://clawhub.ai/api/v1/download?slug=...&tag=latest`, extracts zip into `data/skills/{name}/`.

---

## Decisions Not in Scope (deferred)

- Creating new skills from the UI (phone textarea is too painful)
- Secrets / env var management per skill
- Renaming skills or supporting files
- Adding new supporting files from the UI
