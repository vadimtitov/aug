"""Benchmark fetch_page's tiers on real pages — a regression check, not a unit test.

Runs the real tier logic (fetch_url) on each URL in a file and prints, per URL, the tier
that answered, the classification, HTTP status, elapsed time, extracted length and title,
then a summary.  Uses the stealth browser at $STEALTH_BROWSER_URL; unset, the cheap tier
alone runs (and blocked pages are reported as such).

    STEALTH_BROWSER_URL=http://127.0.0.1:8100 .venv/bin/python scripts/fetch_bench.py \
        scripts/fetch_bench_urls.txt [--concurrency 4] [--json out.json]

Lines starting with # are comments.  Hits the network; costs nothing.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from aug.utils.fetching.fetcher import FetchedPage, fetch_url
from aug.utils.fetching.page_content import Classification
from aug.utils.stealth_browser import StealthBrowserClient


async def main() -> int:
    args = _parse_args()
    urls = [
        line.strip()
        for line in Path(args.urls).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    browser_url = os.environ.get("STEALTH_BROWSER_URL")
    browser = StealthBrowserClient(browser_url) if browser_url else None
    print(f"{len(urls)} urls, stealth browser: {browser_url or 'none'}\n", flush=True)
    print(f"{'tier':8} {'class':13} {'status':>6} {'secs':>6} {'chars':>7}  url — title")

    slots = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()

    async def one(url: str) -> FetchedPage:
        async with slots:
            page = await fetch_url(url, browser)
        print(_row(page), flush=True)
        return page

    pages = await asyncio.gather(*[one(url) for url in urls])
    _summary(pages, time.monotonic() - started)
    if args.json:
        rows = [{**asdict(p), "text": p.text[:500], "text_len": len(p.text)} for p in pages]
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str))
    return 0


# --- Private ---


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("urls", help="file with one URL per line")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--json", help="also write per-URL results here")
    return parser.parse_args()


def _row(page: FetchedPage) -> str:
    title = page.title or (page.note or "")
    return (
        f"{page.tier or '-':8} {page.classification.value:13} {page.status:>6} "
        f"{page.elapsed_s:>6.1f} {len(page.text):>7}  {page.url} — {title[:70]}"
    )


def _summary(pages: list[FetchedPage], wall_s: float) -> None:
    ok = [p for p in pages if p.classification is Classification.OK]
    per_tier = Counter(p.tier for p in ok)
    failed = [p for p in pages if p.classification is not Classification.OK]
    print(
        f"\n{len(ok)}/{len(pages)} ok in {wall_s:.0f}s — "
        + ", ".join(f"{tier}: {n}" for tier, n in sorted(per_tier.items(), key=str))
    )
    for tier in sorted(per_tier, key=str):
        times = sorted(p.elapsed_s for p in ok if p.tier == tier)
        print(f"  {tier}: median {times[len(times) // 2]:.1f}s, max {times[-1]:.1f}s")
    if failed:
        print("failures:")
        for page in failed:
            print(
                f"  {page.classification.value:13} {page.status:>4}  {page.url}  {page.note or ''}"
            )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
