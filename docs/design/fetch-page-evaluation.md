# fetch_page evaluation

How much each layer of `fetch_page` adds when reading real, often bot-protected web pages.
Code: `aug/core/tools/fetch_page/tool.py`, `aug/core/tools/fetch_page/page_content.py`,
`stealth-browser/server.py`.

## Setup

- **Pages:** 384 public URLs across 12 categories, weighted towards commercial sites that
  run bot protection: fashion and retail, marketplaces, travel, food delivery and tickets,
  property and jobs, finance, social, news, docs and dev tools. Landing, category, search
  and product pages. 9 URLs were dead for every method (404/410, no DNS) and are excluded,
  leaving **375**.
- **Success:** the same check for every method. `classify()` says the page is OK: not a
  bot-protection page, not a JavaScript shell, not an HTTP error.
- **Methods**, each run on every URL:
  1. **WebFetch-like:** a plain HTTP request with a `Claude-User` user agent. It
     approximates Claude Code's WebFetch, which can't be run in bulk.
  2. **+ browser headers:** a plain request with Chrome's user agent and `Accept`/`Sec-Fetch`
     headers. The TLS fingerprint is still Python's.
  3. **+ TLS impersonation:** curl_cffi impersonating Chrome and Safari (the cheap tier),
     no browser.
  4. **+ Camoufox:** the full pipeline. It falls back to the stealth-browser sidecar when the
     cheap tier is blocked or gets a shell.
- One run, from one residential IP, with 6 URLs in flight.

## Results

| Method | Pages read | % |
|---|---:|---:|
| 1. WebFetch-like plain request | 175/375 | 47% |
| 2. + browser headers | 260/375 | 69% |
| 3. + TLS impersonation (cheap tier) | 273/375 | 73% |
| **4. + Camoufox (full pipeline)** | **367/375** | **98%** |

Pages gained and lost by each method, against the one before:

| Method | Gained | Lost |
|---|---:|---:|
| + browser headers | +99 | −14 |
| + TLS impersonation | +25 | −12 |
| + Camoufox | +94 | 0 |

- **Headers** fix the most obvious bot filters. Some sites, though, accept an honest bot and
  reject a Python client claiming to be Chrome.
- **TLS impersonation** alone adds little. The sites that check fingerprints mostly also
  run JavaScript challenges.
- **The browser** accounts for nearly all the remaining gain and loses nothing, because it
  only runs when the cheap tier failed.

By category, pages read with each method (1 / 2 / 3 / 4):

| Category | Pages | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Clothing brands | 24 | 5 | 14 | 14 | 24 |
| Fashion | 34 | 15 | 28 | 29 | 33 |
| Retail and marketplaces | 79 | 31 | 48 | 57 | 75 |
| Travel | 48 | 19 | 27 | 29 | 48 |
| Food, tickets, local | 29 | 12 | 21 | 23 | 29 |
| Property, jobs, finance | 41 | 21 | 32 | 29 | 38 |
| Social and media | 33 | 10 | 21 | 24 | 33 |
| News | 12 | 4 | 7 | 7 | 12 |
| Tech and services | 25 | 13 | 19 | 18 | 25 |
| Docs and dev | 45 | 40 | 38 | 38 | 45 |
| Government | 5 | 5 | 5 | 5 | 5 |

Why the WebFetch-like request failed (200 pages):

- 145 bot-protection responses, 122 of them 403;
- 29 JavaScript shells;
- 20 network errors (connections dropped or reset);
- 6 other HTTP errors.

## Latency

| Tier that answered | Share of pages | Median | p90 |
|---|---:|---:|---:|
| Cheap (curl_cffi) | 75% | 0.5 s | 2.7 s |
| Browser (Camoufox) | 25% | 7.8 s | 21 s |

## What still fails

- **Blocked even in the browser:** 4 pages, on Cloudflare, Akamai and AWS WAF.
- **Too little text even in a browser:** 2 pages.
- **The site's own TLS or certificate errors:** 2 pages.

About 8 pages counted as read are of little use: login walls, a country picker shown to
foreign visitors, near-empty single-page apps. No bot-protection page was counted as read.

## Found by the evaluation

A first run showed that block pages often carry a few hundred characters of text. Examples
are a reference number with a language list, or a localised "I am not a robot" notice. A
rule that treated 401/403/429/503 with over 400 characters as a genuine error missed them.
The shipped rule treats a blocking status as bot protection unless 3,000 or more visible
characters come with it, and recognises three more block-page phrasings.

## Limits

- One run from one IP. Bot protection varies by IP reputation, region and time, so the
  numbers are indicative.
- Method 1 approximates WebFetch's request. It is not the tool itself.
- "Read" means the page passed `classify()`. Whether the extracted text answers a given
  question was not judged.
- `scripts/fetch_bench.py` reruns the tiers on any URL list.
