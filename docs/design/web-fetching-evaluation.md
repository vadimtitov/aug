# Web fetching evaluation

Two evaluations of how AUG reads web pages:

1. **Getting the page.** Does the fetching pipeline get a page's content at all, including
   from sites with bot protection? This measures what each layer of the fetcher adds
   (`aug/utils/fetching/`, `stealth-browser/`).
2. **Answering like WebFetch.** Does `web_fetch`, which copies Claude Code's WebFetch on top
   of that pipeline, give the same answers as WebFetch for the same question?

Each was a single run from one residential IP, so the numbers are indicative.

---

## 1. Getting the page: bot protection

How much each layer of the fetcher adds when reading real, often bot-protected pages.

### Setup

- **Pages:** 384 public URLs across 12 categories, weighted towards commercial sites that
  run bot protection: fashion and retail, marketplaces, travel, food delivery and tickets,
  property and jobs, finance, social, news, docs and dev tools. Landing, category, search
  and product pages. 9 URLs were dead for every method (404/410, no DNS) and are excluded,
  leaving **375**.
- **Success:** the same check for every method. `classify()` says the page is OK: not a
  bot-protection page, not a JavaScript shell, not an HTTP error.
- **Methods**, each run on every URL:
  1. **WebFetch-like:** a plain HTTP request with a `Claude-User` user agent. It
     approximates WebFetch's request, which can't be run in bulk.
  2. **+ browser headers:** a plain request with Chrome's user agent and `Accept`/`Sec-Fetch`
     headers. The TLS fingerprint is still Python's.
  3. **+ TLS impersonation:** curl_cffi impersonating Chrome and Safari (the cheap tier),
     no browser.
  4. **+ Camoufox:** the full pipeline. It falls back to the stealth-browser sidecar when the
     cheap tier is blocked or gets a shell.
- 6 URLs in flight at a time.

### Results

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

Why the WebFetch-like request failed (200 pages): 145 bot-protection responses (122 of them
403), 29 JavaScript shells, 20 network errors, 6 other HTTP errors.

### Latency

| Tier that answered | Share of pages | Median | p90 |
|---|---:|---:|---:|
| Cheap (curl_cffi) | 75% | 0.5 s | 2.7 s |
| Browser (Camoufox) | 25% | 7.8 s | 21 s |

### What still fails

- **Blocked even in the browser:** 4 pages, on Cloudflare, Akamai and AWS WAF.
- **Too little text even in a browser:** 2 pages.
- **The site's own TLS or certificate errors:** 2 pages.

About 8 pages counted as read are of little use: login walls, a country picker shown to
foreign visitors, near-empty single-page apps. No bot-protection page was counted as read.

A first run showed that block pages often carry a few hundred characters of text (a
reference number with a language list, a localised "I am not a robot" notice). The shipped
rule treats a blocking status as bot protection unless 3,000 or more visible characters come
with it, and recognises three more block-page phrasings.

### Limits

- Method 1 approximates WebFetch's request; it is not the tool itself.
- "Read" means the page passed `classify()`. Whether the text answers a question is
  section 2's subject.
- `scripts/fetch_bench.py` reruns the tiers on any URL list.

---

## 2. Answering like WebFetch: `web_fetch` against Claude Code's WebFetch

`web_fetch` (`aug/core/tools/web_fetch.py`) copies WebFetch's contract: the agent passes a
URL and a question, a small model reads the page and answers, and only the answer reaches
the agent.

### What was compared

| | Claude Code WebFetch | `web_fetch` |
|---|---|---|
| Fetching | plain request | curl_cffi, then the Camoufox sidecar |
| Page to text | whole page to markdown (Turndown) | whole page to markdown (markdownify) |
| Read per call | 100,000 characters | 100,000 characters |
| Prompt to the small model | `Web page content: --- … --- {question}` + instructions | the same layout; instructions: "Provide a concise response based only on the content above." (WebFetch's quoting and copyright lines left out) |
| Small model | Claude Code's | Claude Haiku (the harness's fast tier) |
| Long page | note with the offset to read on | the same |
| Redirect to another host | not followed, reported | the same |

### Setup

- **37 questions**, each a URL and a prompt, across docs, long reference pages, lists and
  recipes, tables, shops and booking sites (several bot-protected), questions whose answer
  is not on the page, and a cross-host redirect.
- **186 reference facts** written from the full page text: what a correct answer must
  contain (for example, all 8 bank holidays of a year with their dates).
- Each answer was scored per fact (found / partly / missing) by a separate model given
  only the facts and the answer. Made-up facts were counted by hand.

### Results

| | WebFetch | `web_fetch` |
|---|---:|---:|
| Pages read | 31/36 | **36/36** |
| Facts found, all questions | 139/186 (75%) | **169.5/186 (91%)** |
| Facts found, on the 30 pages both read | 90.3% | 89.9% |
| — pages under 100k characters (23) | 99% | 97% |
| — pages over 100k characters (7) | 55% | 61% |
| Answers with a made-up fact | 2 | 1, labelled as general knowledge |
| Cross-host redirect | reported | reported, same format |

The redirect question is not counted in "Pages read".

- **Where both read the page, the answers are equivalent.** `web_fetch`'s are a little
  terser and occasionally drop a secondary detail.
- **WebFetch could not read 5 pages:** two bot-protection errors (403, 503), one empty page
  and two sites it refuses to fetch at all. `web_fetch` read all five, with every reference
  fact on them.
- **Long pages lose facts past 100,000 characters in both.** Which ones differs: the two
  converters produce slightly different markdown, so the window ends at different points
  (`web_fetch`'s longer navigation menu cost it two pages, WebFetch's cut lost another).
- **Made-up facts come from the small model, not the fetching.** Asked for a default the page
  doesn't state, WebFetch's model gave one; `web_fetch`'s gave the same value when it read
  the whole page.
- **Docs domains:** for a few documentation sites that serve markdown, WebFetch skips the
  model and returns the raw page (about 20,000 characters here). `web_fetch` always answers.
- `web_fetch` median latency 3.3 s, p90 5.9 s.

### Page sizes and the 100,000-character limit

The whole-page markdown of the 367 pages section 1 read:

| | Characters |
|---|---:|
| Median | 30,136 |
| p90 | 108,782 |
| p99 | 343,754 |

| Limit | Pages longer than the limit |
|---|---:|
| 100,000 | 10.9% |
| 200,000 | 3.3% |
| 400,000 | 0.8% |

(The main-content extraction alone is far shorter: median 4,853, p90 22,216.)

With a 400,000 limit, `web_fetch` answered all four questions both tools missed at 100,000.
Whether a larger window degrades answers about facts that already fit in 100,000 was not
measured, so `web_fetch` keeps WebFetch's 100,000; a longer page ends with the offset to
read on from.

### Found by the evaluation

The main-content extraction (trafilatura, which `fetch_page` uses) drops content a small
model needs: on a government page it dropped a table's date column, and on a Python docs
page it scrambled inline code. Whole-page markdown kept both, so `web_fetch` converts the
whole page.

### Limits

- 37 questions, one run each; the reference facts were written by hand.
- Bot protection varies by IP and time; WebFetch's failures may differ elsewhere.
