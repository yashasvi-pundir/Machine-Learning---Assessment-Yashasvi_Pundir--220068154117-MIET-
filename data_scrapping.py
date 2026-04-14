import re
import csv
import time
import logging
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_URL      = "https://www.lexaloffle.com"
REQUEST_DELAY = 1.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
}

CSV_FIELDS = ["name", "author", "artwork_url", "game_code", "license", "like_count", "description", "top5_comments"]

SKIP_NAMES = {"", "pico-8", "voxatron", "picotron", "bbs", "faq", "forum", "carts",
              "resources", "schools", "submit", "superblog", "blog"}


def get(session, url, retries=3, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=25, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed – %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(REQUEST_DELAY * attempt * 2)
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries")


def fetch_listing_html(session, page=1):
    params = {"cat": "7", "sub": "2", "mode": "carts", "carts_tab": "1", "page": str(page)}
    resp = get(session, f"{BASE_URL}/bbs/", params=params)
    return resp.text


def parse_listing_html(html):
    """
    The listing page loads cart artwork via JS, so images are NOT in the static HTML.
    We extract: tid, name, detail_url from the <a href="?tid=NNNNN"> link text.
    Everything else is fetched from each cart's detail page.
    """
    soup = BeautifulSoup(html, "lxml")
    entries = []
    seen_tids = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"[?&]tid=(\d+)", href)
        if not m or "#" in href:
            continue

        tid = m.group(1)
        if tid in seen_tids:
            continue

        name = a.get_text(strip=True)
        if not name or name.lower() in SKIP_NAMES:
            continue

        seen_tids.add(tid)
        detail_url = urljoin(BASE_URL, f"/bbs/?tid={tid}")

        entries.append({
            "tid":        tid,
            "name":       name,
            "detail_url": detail_url,
        })

    return entries


def parse_detail_page(html, base_entry):
    soup  = BeautifulSoup(html, "lxml")
    entry = base_entry.copy()

    if not entry.get("name"):
        title_tag = soup.find("title")
        if title_tag:
            raw = title_tag.get_text(strip=True)
            entry["name"] = re.split(r"[|\-–]", raw)[0].strip()

    entry["author"] = ""
    for a in soup.find_all("a", href=True):
        if "uid=" in a["href"]:
            txt = a.get_text(strip=True)
            if txt and txt.lower() not in SKIP_NAMES:
                entry["author"] = txt
                break

    entry["artwork_url"] = ""
    
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if any(x in src for x in ["/bbs/cposts", "/bbs/gfx", "label", ".p8.png"]):
            if not any(x in src.lower() for x in ["logo", "icon", "avatar", "/gfx/"]):
                entry["artwork_url"] = urljoin(BASE_URL, src)
                break

    
    if not entry["artwork_url"]:
        for img in soup.find_all("img", src=True):
            src = img["src"]
            if src and "/gfx/" not in src and src.endswith((".png", ".jpg", ".gif")):
                entry["artwork_url"] = urljoin(BASE_URL, src)
                break

    entry["like_count"] = 0
    full_text = soup.get_text(" ")
    hm = re.search(r"(\d+)\s*[♥❤]|[♥❤]\s*(\d+)", full_text)
    if hm:
        entry["like_count"] = int(hm.group(1) or hm.group(2))
    if not entry["like_count"]:
        
        for sel in ["[id*='fav']", "[class*='fav']", "[id*='like']", "[class*='like']", "[id*='heart']"]:
            tag = soup.select_one(sel)
            if tag:
                nums = re.findall(r"\d+", tag.get_text())
                if nums:
                    entry["like_count"] = int(nums[0])
                    break

    # ── Description: main post body ──────────────────────────────────────────
    entry["description"] = ""
    # Lexaloffle wraps the post text in a div with id like "bbs_post_body_NNNNN"
    # or a class containing "post_body"
    desc_elem = (
        soup.find("div", id=re.compile(r"bbs_post_body", re.I))
        or soup.find("div", class_=re.compile(r"post.?body", re.I))
        or soup.find("div", id=re.compile(r"post.?body", re.I))
    )
    if desc_elem:
        for tag in desc_elem.find_all(["script", "iframe", "noscript", "style"]):
            tag.decompose()
        entry["description"] = desc_elem.get_text(separator=" ", strip=True)[:2000]

    # ── Game code ────────────────────────────────────────────────────────────
    entry["game_code"] = ""
    # Lexaloffle embeds cart data in a JS call like:
    #   pico8_set_ram(...) or var cart = {...} or pxa={...}
    for scr in soup.find_all("script"):
        txt = scr.get_text()
        for pat in [
            r'pico8_code\s*=\s*["\`](.+?)["\`]\s*;',
            r'"code"\s*:\s*"(.+?)"',
            r"'code'\s*:\s*'(.+?)'",
            r'cart_code\s*=\s*["\`](.+?)["\`]',
            r'var\s+code\s*=\s*["\`](.+?)["\`]',
        ]:
            m = re.search(pat, txt, re.S)
            if m:
                entry["game_code"] = m.group(1)[:5000]
                break
        if entry["game_code"]:
            break

    # Fallback: iframe src or direct .p8 link
    if not entry["game_code"]:
        iframe = soup.find("iframe", src=re.compile(r"\.p8|pico|lexaloffle", re.I))
        if iframe and iframe.get("src"):
            entry["game_code"] = urljoin(BASE_URL, iframe["src"])

    if not entry["game_code"]:
        p8_a = soup.find("a", href=re.compile(r"\.p8(\.png)?(\?|$)", re.I))
        if p8_a:
            entry["game_code"] = urljoin(BASE_URL, p8_a["href"])

    # ── License ──────────────────────────────────────────────────────────────
    entry["license"] = ""
    for pat in [
        re.compile(r"licen[sc]e\s*[:\-]?\s*([^\n\.]{3,80})", re.I),
        re.compile(r"\b(CC[- ](?:BY|BY-SA|BY-NC|BY-ND|BY-NC-SA|BY-NC-ND)(?:[- ]\d\.\d)?)\b", re.I),
        re.compile(r"\b(MIT(?: License)?)\b", re.I),
        re.compile(r"\b(GPL[\s-]?v?\d?|GNU General Public License)\b", re.I),
        re.compile(r"\b(Apache[\s-]2\.0|Apache License)\b", re.I),
        re.compile(r"\b(public domain|unlicense|WTFPL|zlib)\b", re.I),
    ]:
        m = pat.search(full_text)
        if m:
            entry["license"] = m.group(1).strip()[:120]
            break

    # ── Top-5 comments ───────────────────────────────────────────────────────
    # On Lexaloffle detail pages, each reply is a <div id="pNNNNNN"> block
    # The FIRST such div is the original post, so we skip it (start from index 1)
    comments       = []
    comment_blocks = soup.find_all("div", id=re.compile(r"^p\d+$"))
    for block in comment_blocks[1:6]:  # skip first (= original post), take next 5
        c_author = "unknown"
        for ua in block.find_all("a", href=True):
            if "uid=" in ua["href"]:
                txt = ua.get_text(strip=True)
                if txt:
                    c_author = txt
                    break

        # Comment body: find nested div with post body content
        c_body_elem = (
            block.find("div", id=re.compile(r"bbs_post_body", re.I))
            or block.find("div", class_=re.compile(r"post.?body", re.I))
        )
        if c_body_elem:
            c_body = c_body_elem.get_text(separator=" ", strip=True)
        else:
            # Fallback: get all text from the block, strip username/metadata noise
            c_body = block.get_text(separator=" ", strip=True)

        c_body = re.sub(r"\s+", " ", c_body).strip()[:300].replace("|", "/")
        if c_body:
            comments.append(f"[{c_author}]: {c_body}")

    entry["top5_comments"] = " | ".join(comments)
    return entry


def scrape(limit=100, output="lexaloffle_games.csv"):
    session     = requests.Session()
    all_entries = []
    page        = 1

    log.info("Starting scrape – target: %d games", limit)

    while len(all_entries) < limit:
        log.info("Fetching listing page %d…", page)
        html    = fetch_listing_html(session, page=page)
        entries = parse_listing_html(html)

        if not entries:
            log.warning("No entries on page %d – stopping.", page)
            break

        existing_tids = {e.get("tid") for e in all_entries}
        new_entries   = [e for e in entries if e.get("tid") not in existing_tids]
        all_entries.extend(new_entries)
        log.info("  +%d carts (total: %d)", len(new_entries), len(all_entries))

        if len(all_entries) >= limit:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    all_entries = all_entries[:limit]
    log.info("Fetching %d detail pages…", len(all_entries))

    enriched = []
    for idx, entry in enumerate(all_entries, 1):
        detail_url = entry.get("detail_url", "")
        log.info("[%d/%d] %s", idx, len(all_entries), detail_url)
        try:
            resp       = get(session, detail_url)
            full_entry = parse_detail_page(resp.text, entry)
        except Exception as exc:
            log.error("  Skipping – %s", exc)
            full_entry = entry

        for field in CSV_FIELDS:
            full_entry.setdefault(field, "")

        log.info("  %-35s | author=%-18s | likes=%-4s | artwork=%s | code=%s",
                 repr(full_entry.get("name","")[:33]),
                 repr(full_entry.get("author","")[:16]),
                 full_entry.get("like_count","?"),
                 "YES" if full_entry.get("artwork_url") else "NO ",
                 "YES" if full_entry.get("game_code")   else "NO")

        enriched.append(full_entry)
        time.sleep(REQUEST_DELAY)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    log.info("Done. Saved %d games → %s", len(enriched), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PICO-8 carts from Lexaloffle BBS")
    parser.add_argument("--limit",  type=int, default=100,                    help="Number of games (default: 100)")
    parser.add_argument("--output", type=str, default="lexaloffle_games.csv", help="Output CSV file")
    args = parser.parse_args()
    scrape(limit=args.limit, output=args.output)
