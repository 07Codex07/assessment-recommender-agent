
"""
SHL Catalog Scraper - Run locally to generate shl_catalog.json
pip install playwright && python -m playwright install chromium && python scraper.py
"""
import json, time, re
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
TEST_TYPE_MAP = {"A":"Ability & Aptitude","B":"Biodata & Situational Judgement","C":"Competencies","D":"Development & 360","E":"Assessment Exercises","K":"Knowledge & Skills","P":"Personality & Behavior","S":"Simulations"}

def scrape_page(page, start):
    url = f"{CATALOG_URL}?start={start}&type=1&action_doFilteringForm=Search"
    page.goto(url, wait_until="networkidle", timeout=30000)
    page.wait_for_timeout(2000)
    products = []
    for row in page.query_selector_all("table tbody tr"):
        tds = row.query_selector_all("td")
        if len(tds) < 4: continue
        link = tds[0].query_selector("a")
        if not link: continue
        name = link.inner_text().strip()
        href = link.get_attribute("href") or ""
        if not name or not href: continue
        prod_url = href if href.startswith("http") else BASE_URL + href
        has = lambda td: bool(td.inner_text().strip()) and td.inner_text().strip() not in ["-",""]
        type_text = tds[3].inner_text().strip()
        test_types = [c for c in type_text if c in TEST_TYPE_MAP]
        products.append({"name":name,"url":prod_url,"remote_testing":has(tds[1]),"adaptive_irt":has(tds[2]),"test_types":test_types,"test_type_labels":[TEST_TYPE_MAP[t] for t in test_types],"description":"","job_levels":[]})
    return products

def get_max_start(page):
    page.goto(f"{CATALOG_URL}?start=0&type=1&action_doFilteringForm=Search", wait_until="networkidle", timeout=30000)
    page.wait_for_timeout(2000)
    max_s = 0
    for link in page.query_selector_all("a"):
        m = re.search(r"start=(\d+)", link.get_attribute("href") or "")
        if m: max_s = max(max_s, int(m.group(1)))
    return max_s

def scrape_detail(page, url):
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
        page.wait_for_timeout(1000)
        desc = ""
        for sel in ["main p",".product-hero p","p"]:
            el = page.query_selector(sel)
            if el:
                t = el.inner_text().strip()
                if len(t) > 40: desc = t[:600]; break
        full = page.inner_text("body")
        levels = [jl for jl in ["Director","Entry-Level","Executive","Front Line Manager","General Population","Graduate","Manager","Mid-Professional","Professional Individual Contributor","Supervisor"] if jl in full]
        return {"description":desc,"job_levels":levels}
    except: return {"description":"","job_levels":[]}

def main():
    products = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(user_agent="Mozilla/5.0").new_page()
        max_start = get_max_start(page)
        print(f"Max start: {max_start}")
        for i, start in enumerate(range(0, max_start+12, 12)):
            print(f"Page {i+1} (start={start})")
            try:
                prods = scrape_page(page, start)
                products.extend(prods)
                time.sleep(1.5)
            except Exception as e: print(f"Error: {e}")
        seen = set()
        unique = [p for p in products if p["url"] not in seen and not seen.add(p["url"])]
        print(f"Unique: {len(unique)}, scraping details...")
        for i, prod in enumerate(unique):
            if i%20==0: print(f"Detail {i}/{len(unique)}")
            prod.update(scrape_detail(page, prod["url"]))
            time.sleep(0.8)
        browser.close()
    with open("shl_catalog.json","w") as f:
        json.dump({"scraped_at":time.strftime("%Y-%m-%d"),"total":len(unique),"products":unique},f,indent=2)
    print(f"Saved {len(unique)} products")

if __name__ == "__main__": main()
