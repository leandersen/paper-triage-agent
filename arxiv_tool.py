"""Tool functions for agent to call."""
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Retry on transient network failures. We treat TimeoutError and URLError as
# "the server might just be slow"; everything else (e.g. ValueError in our
# own code) is a real bug and should NOT be retried.
NETWORK_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, urllib.error.URLError)),
    reraise=True,
    before_sleep=lambda retry_state: print(
        f"NOTICE: arXiv call failed (attempt {retry_state.attempt_number}), "
        f"retrying in {retry_state.next_action.sleep:.1f}s..."
    ),
)

@NETWORK_RETRY
def search_arxiv(query: str, max_results: int=5) -> list[dict]:
    """Search arXiv and return list of papers. 
    Returns list of dicts with keys: id, title, authors, summary, published.
    """
    base_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy":"relevance",
        "sortOrder":"descending",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url, timeout=30) as response:
        xml_data = response.read().decode('utf-8')
    
    # arxiv returns atom xml
    ns = {'atom':'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(xml_data)

    papers = []
    for entry in root.findall("atom:entry", ns):
        paper = {
            'id':entry.find('atom:id', ns).text.split('/')[-1],
            'title':entry.find('atom:title',ns).text.strip().replace('\n',' '),
            'authors': [
                a.find('atom:name', ns).text
                for a in entry.findall('atom:author',ns)
            ],
            'summary':entry.find('atom:summary',ns).text.strip().replace('\n', ' '),
            'published':entry.find('atom:published', ns).text[:10]
        }
        papers.append(paper)
    return papers

if __name__ == '__main__':
    results = search_arxiv('retrieval augmented generation', max_results = 2)
    for p in results:
        print(f"-   {p['title']} ({p['published']})")
        print(f"    {p['id']}")