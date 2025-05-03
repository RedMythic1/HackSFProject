from duckduckgo_search import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text("how to build a car", max_results=5))
    for r in results:
        print(r["title"], r["href"]) 