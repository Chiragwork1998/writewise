"""Firecrawl fetch layer — every page's content comes through Firecrawl.

- Spreads work across several API keys, each with its own concurrency limit.
- Honours robots.txt disallow rules and crawl-delay per host (from discovery/sitemaps/*.json).
- Saves every result to pages/<sha1>.json immediately (Firecrawl drops results after 24h).
- Appends one line per request to logs/fetch_log.jsonl, including credits used.
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

API = os.environ.get("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v2")  # set to a self-hosted instance to switch


def load_env(path):
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def url_id(url):
    return hashlib.sha1(url.encode()).hexdigest()


def normalize(url):
    url = url.strip().split("#", 1)[0]
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


class Robots:
    def __init__(self, sitemaps_dir):
        self.rules = {}
        for f in Path(sitemaps_dir).glob("*.json"):
            try:
                rec = json.loads(f.read_text())
                self.rules[rec["site"]] = rec["robots"]
            except Exception:
                pass

    @staticmethod
    def _match(pattern, path):
        rx = "^" + re.escape(pattern).replace(r"\*", ".*")
        if rx.endswith(r"\$"):
            rx = rx[:-2] + "$"
        return re.match(rx, path) is not None

    def allowed(self, url):
        p = urlparse(url)
        r = self.rules.get(p.hostname)
        if not r:
            return True
        path = p.path + (("?" + p.query) if p.query else "")
        best_allow = max((len(a) for a in r["allow"] if self._match(a, path)), default=-1)
        best_dis = max((len(d) for d in r["disallow"] if self._match(d, path)), default=-1)
        return best_allow >= best_dis

    def delay(self, host):
        r = self.rules.get(host)
        return (r or {}).get("delay") or 0


class Fetcher:
    def __init__(self, college_dir, keys, default_host_gap=1.0):
        """keys: list of (api_key, concurrency, requests_per_minute)."""
        self.college = Path(college_dir)
        self.pages = self.college / "pages"
        self.pages.mkdir(exist_ok=True)
        self.log_path = self.college / "logs" / "fetch_log.jsonl"
        self.robots = Robots(self.college / "discovery" / "sitemaps")
        self.default_gap = default_host_gap
        self.key_slots = []
        self.key_rpm = {}
        self.key_calls = {}
        for k, n, rpm in keys:
            self.key_slots.append((k, threading.Semaphore(n)))
            self.key_rpm[k] = rpm
            self.key_calls[k] = []
        self.host_next = {}
        self.host_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.rr = 0

    def done(self, url):
        return (self.pages / f"{url_id(url)}.json").exists()

    def _wait_host(self, host):
        gap = max(self.robots.delay(host), self.default_gap)
        while True:
            with self.host_lock:
                now = time.time()
                nxt = self.host_next.get(host, 0)
                if now >= nxt:
                    self.host_next[host] = now + gap
                    return
            time.sleep(min(nxt - now, 5))

    def _acquire_key(self):
        """Pick a key with a free concurrency slot and room under its per-minute request limit."""
        while True:
            now = time.time()
            with self.host_lock:
                for i in range(len(self.key_slots)):
                    k, sem = self.key_slots[(self.rr + i) % len(self.key_slots)]
                    calls = [t for t in self.key_calls[k] if now - t < 60]
                    self.key_calls[k] = calls
                    if len(calls) >= self.key_rpm[k]:
                        continue
                    if sem.acquire(blocking=False):
                        calls.append(now)
                        self.rr += 1
                        return k, sem
            time.sleep(0.25)

    def scrape(self, url, meta=None, formats=("markdown", "links"), only_main=False, pdf_max_pages=12,
               wait_for=0, timeout_ms=90000, retries=2, actions=None):
        url = normalize(url)
        out = self.pages / f"{url_id(url)}.json"
        if out.exists():
            return json.loads(out.read_text())
        if not self.robots.allowed(url):
            rec = {"url": url, "status": "robots_disallowed", "meta": meta or {}}
            self._log(rec, 0, None)
            return rec
        host = urlparse(url).hostname
        body = {
            "url": url,
            "formats": list(formats),
            "onlyMainContent": only_main,
            "parsers": [{"type": "pdf", "maxPages": pdf_max_pages}],
            "timeout": timeout_ms,
            "blockAds": True,
            "removeBase64Images": True,
        }
        if wait_for:
            body["waitFor"] = wait_for
        if actions:
            body["actions"] = actions
            body["timeout"] = max(timeout_ms, 150000)
        attempt = 0
        while True:
            self._wait_host(host)
            key, sem = self._acquire_key()
            t0 = time.time()
            try:
                r = httpx.post(f"{API}/scrape", json=body, timeout=body["timeout"] / 1000 + 90,
                               headers={"Authorization": f"Bearer {key}"})
                code = r.status_code
                j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            except Exception as e:
                code, j = 0, {"error": f"{type(e).__name__}: {e}"}
            finally:
                sem.release()
            elapsed = round(time.time() - t0, 1)
            if code == 200 and j.get("success"):
                data = j.get("data", {})
                md = data.get("metadata", {}) or {}
                rec = {
                    "url": url,
                    "status": "ok",
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "http_status": md.get("statusCode"),
                    "final_url": md.get("url") or md.get("sourceURL") or url,
                    "title": md.get("title") or md.get("ogTitle"),
                    "description": md.get("description") or md.get("ogDescription"),
                    "published": md.get("publishedTime") or md.get("article:published_time") or md.get("dcDate"),
                    "modified": md.get("modifiedTime") or md.get("article:modified_time"),
                    "language": md.get("language"),
                    "content_type": md.get("contentType"),
                    "credits": md.get("creditsUsed"),
                    "markdown": data.get("markdown") or "",
                    "raw_html": data.get("rawHtml") or "",
                    "links": data.get("links") or [],
                    "metadata": md,
                    "meta": meta or {},
                    "key": key[-4:],
                    "elapsed": elapsed,
                }
                out.write_text(json.dumps(rec))
                self._log(rec, rec["credits"], code)
                return rec
            err_text = str(j.get("error", "")) if isinstance(j, dict) else ""
            if ("All scraping engines failed" in err_text or "not long enough" in err_text) and not body.get("waitFor") \
                    and "api.firecrawl.dev" not in API and not actions:
                body["waitFor"] = 8000  # self-hosted: JavaScript apps can render after the first capture
                continue
            retryable = code in (0, 408, 429, 500, 502, 503, 504)
            if code == 402:  # out of credits on this key — drop it
                self.key_slots = [(k, s) for k, s in self.key_slots if k != key]
                if not self.key_slots:
                    rec = {"url": url, "status": "no_credits", "meta": meta or {}}
                    self._log(rec, 0, code)
                    return rec
                continue
            if retryable and attempt < retries:
                attempt += 1
                time.sleep(5 * attempt + (30 if code == 429 else 0))
                continue
            rec = {"url": url, "status": "error", "http": code, "error": str(j.get("error", ""))[:500],
                   "meta": meta or {}, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self._log(rec, j.get("creditsUsed", 0) if isinstance(j, dict) else 0, code)
            return rec

    def _log(self, rec, credits, code):
        line = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": rec["url"], "status": rec["status"],
                "api_http": code, "credits": credits, "chars": len(rec.get("markdown", "") or ""),
                "bucket": (rec.get("meta") or {}).get("bucket")}
        with self.log_lock:
            with self.log_path.open("a") as f:
                f.write(json.dumps(line) + "\n")

    def map(self, url, limit=100000, search=None):
        body = {"url": url, "limit": limit, "includeSubdomains": False, "sitemap": "include"}
        if search:
            body["search"] = search
        key, sem = self._acquire_key()
        try:
            r = httpx.post(f"{API}/map", json=body, timeout=300, headers={"Authorization": f"Bearer {key}"})
        finally:
            sem.release()
        j = r.json()
        with self.log_lock:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": url,
                                    "status": "map", "api_http": r.status_code, "credits": 1,
                                    "links": len(j.get("links", []))}) + "\n")
        return j

    def search(self, query, limit=20, tbs=None):
        body = {"query": query, "limit": limit}
        if tbs:
            body["tbs"] = tbs
        key, sem = self._acquire_key()
        try:
            r = httpx.post(f"{API}/search", json=body, timeout=120, headers={"Authorization": f"Bearer {key}"})
        finally:
            sem.release()
        j = r.json()
        with self.log_lock:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": query,
                                    "status": "search", "api_http": r.status_code,
                                    "credits": j.get("creditsUsed", -(-limit // 10) * 2)}) + "\n")
        return j


def default_fetcher(college_dir):
    root = Path(__file__).resolve().parent.parent
    load_env(root / ".env")
    if os.environ.get("FIRECRAWL_SELFHOST"):
        assert "api.firecrawl.dev" not in API, "FIRECRAWL_SELFHOST set but FIRECRAWL_API_URL points at the cloud"
        n = int(os.environ.get("FIRECRAWL_CONCURRENCY", "10"))
        return Fetcher(college_dir, [("self-hosted", n, 100000)])
    # Hobby plan: 5 concurrent, 100 req/min. Free plan: 2 concurrent, 10 req/min (kept a notch under).
    keys = [(os.environ["FIRECRAWL_API_KEY"], 5, 90)]
    if os.environ.get("FIRECRAWL_API_KEY_2"):
        keys.append((os.environ["FIRECRAWL_API_KEY_2"], 2, 8))
    return Fetcher(college_dir, keys)


# Clicks every control whose text is exactly `text` (e.g. "Expand All" on a schedule of classes) so hidden sections render.
def expand_actions(text="Expand All"):
    return [
        {"type": "wait", "milliseconds": 4000},
        {"type": "executeJavascript", "script": """(() => {
  const els=[...document.querySelectorAll('*')].filter(e=>e.children.length<=3 && e.textContent.trim()===%s);
  const out=[];
  for (const e of els) {
    const sw = e.closest('mat-slide-toggle, .mat-mdc-slide-toggle, [class*="toggle"], [class*="switch"]');
    const target = sw ? (sw.querySelector('button[role="switch"], input, button') || sw) : e;
    if (!out.includes(target)) { target.click(); out.push(target); }
  }
  return out.length;
})()""" % json.dumps(text).replace('"', "'")},
        {"type": "wait", "milliseconds": 7000},
    ]


EXPAND_ALL_ACTIONS = expand_actions()
