from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import urllib.parse
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Gemini API key
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

# FastAPI app
app = FastAPI(title="Enhanced Academic Paper Search API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ==== Data Models ====
class PromptRequest(BaseModel):
    query: str

class PaperCard(BaseModel):
    title: str
    authors: str
    snippet: str
    link: str
    source: str = "Unknown"

# ==== Helper Functions ====
def extract_clean_link(title_tag) -> str:
    """Extract clean link from Google Scholar result"""
    if title_tag and title_tag.a:
        href = title_tag.a.get("href", "")
        if href.startswith("/scholar?"):
            return f"https://scholar.google.com{href}"
        elif href.startswith("http"):
            return href
    return "No link available"

def get_random_user_agent() -> str:
    """Get random user agent to avoid blocking"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0",
    ]
    return random.choice(user_agents)

def create_session_with_retries() -> requests.Session:
    """Create requests session with retry strategy"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# ==== Gemini Functions ====
def generate_search_query(user_prompt: str) -> str:
    """Generate optimized search query using Gemini"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Convert the following user intent into a concise academic search query.
        Return only the search query without any additional text or formatting:
        
        User query: {user_prompt}
        
        Academic search query:"""
        
        response = model.generate_content(prompt)
        return response.text.strip().strip('"').strip("'")
    except Exception as e:
        logger.error(f"Error generating search query: {e}")
        return user_prompt

def summarize_snippet(snippet: str) -> str:
    """Summarize long snippets using Gemini"""
    try:
        if len(snippet) < 100:
            return snippet
            
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""Summarize the following academic paper snippet in one clear, concise sentence.
        Focus on the main finding or contribution. Return only the summary:
        
        Snippet: {snippet}
        
        Summary:"""
        
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        if len(summary) < 20:
            return snippet
        return summary
    except Exception as e:
        logger.error(f"Error summarizing snippet: {e}")
        return snippet

# ==== Multi-Source Search Functions ====
def search_semantic_scholar(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search papers using Semantic Scholar API"""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,url,year,venue,citationCount,externalIds"
        }
        
        headers = {"User-Agent": get_random_user_agent()}
        session = create_session_with_retries()
        response = session.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        papers = data.get("data", [])
        
        results = []
        for paper in papers:
            paper_url = paper.get("url", "")
            if not paper_url:
                external_ids = paper.get("externalIds", {})
                if external_ids.get("DOI"):
                    paper_url = f"https://doi.org/{external_ids['DOI']}"
                elif external_ids.get("ArXiv"):
                    paper_url = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
                else:
                    paper_url = "No direct link available"
            
            results.append({
                "title": paper.get("title", "No title available"),
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "url": paper_url,
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "citation_count": paper.get("citationCount", 0),
                "source": "Semantic Scholar"
            })
        
        logger.info(f"Semantic Scholar returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Semantic Scholar search error: {e}")
        return []

def search_arxiv(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search papers using ArXiv API"""
    try:
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        headers = {"User-Agent": get_random_user_agent()}
        session = create_session_with_retries()
        response = session.get(base_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        results = []
        entries = root.findall('atom:entry', ns)
        
        for entry in entries:
            title = entry.find('atom:title', ns)
            title_text = title.text.strip().replace('\n', ' ') if title is not None and title.text else "No title available"
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            summary = entry.find('atom:summary', ns)
            summary_text = summary.text.strip() if summary is not None and summary.text else "No abstract available"
            
            id_elem = entry.find('atom:id', ns)
            url = id_elem.text if id_elem is not None else "No link available"
            
            published = entry.find('atom:published', ns)
            year = None
            if published is not None and published.text:
                try:
                    year = int(published.text[:4])
                except:
                    year = None
            
            results.append({
                "title": title_text,
                "authors": [{"name": author} for author in authors],
                "abstract": summary_text,
                "url": url,
                "year": year,
                "venue": "arXiv",
                "citation_count": 0,
                "source": "arXiv"
            })
        
        logger.info(f"arXiv returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return []

def search_google_scholar_robust(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Enhanced Google Scholar search with anti-blocking measures"""
    try:
        search_strategies = [
            f"https://scholar.google.com/scholar?q={urllib.parse.quote_plus(query)}&hl=en&num={limit}",
            f"https://scholar.google.co.uk/scholar?q={urllib.parse.quote_plus(query)}&hl=en&num={limit}",
            f"https://scholar.google.ca/scholar?q={urllib.parse.quote_plus(query)}&hl=en&num={limit}",
        ]
        
        for attempt, search_url in enumerate(search_strategies):
            try:
                if attempt > 0:
                    time.sleep(random.uniform(2, 5))
                
                headers = {
                    "User-Agent": get_random_user_agent(),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Referer": "https://scholar.google.com/",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "same-origin",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache"
                }
                
                session = create_session_with_retries()
                response = session.get(search_url, headers=headers, timeout=20)
                
                if response.status_code == 429:
                    logger.warning(f"Rate limited on attempt {attempt + 1}")
                    continue
                
                response.raise_for_status()
                
                if "blocked" in response.text.lower() or "captcha" in response.text.lower():
                    logger.warning(f"Blocked on attempt {attempt + 1}")
                    continue
                
                soup = BeautifulSoup(response.text, "html.parser")
                paper_results = soup.select(".gs_ri")
                
                if not paper_results:
                    continue
                
                results = []
                for result in paper_results[:limit]:
                    try:
                        title_tag = result.select_one(".gs_rt")
                        title = title_tag.get_text(strip=True) if title_tag else "No title available"
                        title = title.replace("[PDF]", "").replace("[HTML]", "").replace("[CITATION]", "").strip()
                        
                        link = extract_clean_link(title_tag)
                        
                        snippet_tag = result.select_one(".gs_rs")
                        abstract = snippet_tag.get_text(strip=True) if snippet_tag else "No abstract available"
                        
                        author_tag = result.select_one(".gs_a")
                        authors_text = author_tag.get_text(strip=True) if author_tag else "No author information"
                        
                        authors = []
                        year = None
                        if authors_text and " - " in authors_text:
                            parts = authors_text.split(" - ")
                            if parts:
                                author_part = parts[0]
                                authors = [{"name": name.strip()} for name in author_part.split(",")]
                                
                                import re
                                for part in parts[1:]:
                                    year_match = re.search(r'\b(19|20)\d{2}\b', part)
                                    if year_match:
                                        year = int(year_match.group())
                                        break
                        
                        results.append({
                            "title": title,
                            "authors": authors,
                            "abstract": abstract,
                            "url": link,
                            "year": year,
                            "venue": "",
                            "citation_count": 0,
                            "source": "Google Scholar"
                        })
                    except Exception as e:
                        logger.error(f"Error processing Google Scholar result: {e}")
                        continue
                
                if results:
                    logger.info(f"Google Scholar returned {len(results)} results")
                    return results
                
            except Exception as e:
                logger.error(f"Google Scholar attempt {attempt + 1} failed: {e}")
                continue
        
        return []
        
    except Exception as e:
        logger.error(f"Google Scholar search error: {e}")
        return []

def search_pubmed_via_ncbi(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search PubMed using NCBI E-utilities"""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json",
            "sort": "relevance"
        }
        
        headers = {"User-Agent": get_random_user_agent()}
        session = create_session_with_retries()
        
        search_response = session.get(search_url, params=search_params, headers=headers, timeout=15)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []
        
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        
        fetch_response = session.get(fetch_url, params=fetch_params, headers=headers, timeout=20)
        fetch_response.raise_for_status()
        
        from xml.etree import ElementTree as ET
        root = ET.fromstring(fetch_response.content)
        
        results = []
        for article in root.findall('.//PubmedArticle'):
            try:
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else "No title available"
                
                authors = []
                author_list = article.findall('.//Author')
                for author in author_list:
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    if last_name is not None:
                        name = last_name.text
                        if first_name is not None:
                            name = f"{first_name.text} {name}"
                        authors.append({"name": name})
                
                abstract_elem = article.find('.//Abstract/AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                
                year_elem = article.find('.//PubDate/Year')
                year = int(year_elem.text) if year_elem is not None and year_elem.text else None
                
                journal_elem = article.find('.//Journal/Title')
                venue = journal_elem.text if journal_elem is not None else ""
                
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ""
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "No link available"
                
                results.append({
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "url": url,
                    "year": year,
                    "venue": venue,
                    "citation_count": 0,
                    "source": "PubMed"
                })
                
            except Exception as e:
                logger.error(f"Error processing PubMed result: {e}")
                continue
        
        logger.info(f"PubMed returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"PubMed search error: {e}")
        return []

def merge_and_deduplicate_results(all_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge results from multiple sources and remove duplicates"""
    merged = []
    seen_titles = set()
    
    for source_results in all_results:
        for result in source_results:
            title = result.get("title", "").lower().strip()
            if title and title not in seen_titles and len(title) > 10:
                seen_titles.add(title)
                merged.append(result)
    
    source_priority = {"Semantic Scholar": 1, "PubMed": 2, "arXiv": 3, "Google Scholar": 4}
    merged.sort(key=lambda x: source_priority.get(x.get("source", ""), 5))
    
    return merged

def format_paper_result(paper: Dict[str, Any]) -> PaperCard:
    """Format a paper result into PaperCard format"""
    authors_list = paper.get("authors", [])
    if authors_list:
        author_names = [author.get("name", "Unknown") for author in authors_list]
        authors_str = ", ".join(author_names[:3])
        if len(authors_list) > 3:
            authors_str += " et al."
    else:
        authors_str = "No author information"
    
    year = paper.get("year")
    venue = paper.get("venue")
    citation_count = paper.get("citation_count", 0)
    source = paper.get("source", "Unknown")
    
    if year or venue:
        additional_info = []
        if year:
            additional_info.append(str(year))
        if venue:
            additional_info.append(venue)
        if citation_count > 0:
            additional_info.append(f"Citations: {citation_count}")
        
        authors_str += f" - {', '.join(additional_info)} ({source})"
    else:
        authors_str += f" ({source})"
    
    abstract = paper.get("abstract", "")
    if abstract and len(abstract) > 300:
        snippet = summarize_snippet(abstract)
    else:
        snippet = abstract if abstract else "No abstract available"
    
    return PaperCard(
        title=paper.get("title", "No title available"),
        authors=authors_str,
        snippet=snippet,
        link=paper.get("url", "No link available"),
        source=source
    )

# ==== API Endpoints ====
@app.post("/search", response_model=List[PaperCard])
async def search_papers_multi_source(data: PromptRequest):
    """Enhanced search using multiple academic sources with anti-blocking measures"""
    try:
        if not data.query or not data.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        smart_query = generate_search_query(data.query)
        logger.info(f"Original query: {data.query}")
        logger.info(f"Generated search query: {smart_query}")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            semantic_future = executor.submit(search_semantic_scholar, smart_query, 8)
            arxiv_future = executor.submit(search_arxiv, smart_query, 6)
            pubmed_future = executor.submit(search_pubmed_via_ncbi, smart_query, 6)
            scholar_future = executor.submit(search_google_scholar_robust, smart_query, 8)
            
            all_results = []
            
            try:
                semantic_results = semantic_future.result(timeout=15)
                all_results.append(semantic_results)
                logger.info(f"Semantic Scholar: {len(semantic_results)} results")
            except Exception as e:
                logger.error(f"Semantic Scholar failed: {e}")
                all_results.append([])
            
            try:
                arxiv_results = arxiv_future.result(timeout=15)
                all_results.append(arxiv_results)
                logger.info(f"arXiv: {len(arxiv_results)} results")
            except Exception as e:
                logger.error(f"arXiv failed: {e}")
                all_results.append([])
            
            try:
                pubmed_results = pubmed_future.result(timeout=20)
                all_results.append(pubmed_results)
                logger.info(f"PubMed: {len(pubmed_results)} results")
            except Exception as e:
                logger.error(f"PubMed failed: {e}")
                all_results.append([])
            
            try:
                scholar_results = scholar_future.result(timeout=25)
                all_results.append(scholar_results)
                logger.info(f"Google Scholar: {len(scholar_results)} results")
            except Exception as e:
                logger.error(f"Google Scholar failed: {e}")
                all_results.append([])
        
        merged_results = merge_and_deduplicate_results(all_results)
        
        if not merged_results:
            logger.warning("No results found from any source")
            return []
        
        formatted_results = []
        for idx, paper in enumerate(merged_results[:15]):
            try:
                formatted_paper = format_paper_result(paper)
                formatted_results.append(formatted_paper)
                
                if idx > 0 and len(paper.get("abstract", "")) > 300:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error formatting paper {idx}: {e}")
                continue
        
        total_sources = sum(1 for results in all_results if results)
        logger.info(f"Successfully processed {len(formatted_results)} results from {total_sources} sources")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Multi-source search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search-scholar", response_model=List[PaperCard])
async def search_papers_scholar(data: PromptRequest):
    """Backup endpoint using Google Scholar only"""
    try:
        if not data.query or not data.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        smart_query = generate_search_query(data.query)
        results = search_google_scholar_robust(smart_query, 10)
        
        if not results:
            return []
        
        formatted_results = []
        for idx, paper in enumerate(results):
            try:
                formatted_paper = format_paper_result(paper)
                formatted_results.append(formatted_paper)
                
                if idx > 0:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error formatting paper {idx}: {e}")
                continue
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Scholar search error: {e}")
        raise HTTPException(status_code=500, detail="Google Scholar search failed")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Enhanced Academic Paper Search API is running"}

@app.get("/")
async def root():
    return {
        "message": "Enhanced Academic Paper Search API",
        "version": "2.0.0",
        "endpoints": {
            "POST /search": "Search for academic papers (multi-source)",
            "POST /search-scholar": "Search using Google Scholar only (backup)",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "sources": ["Semantic Scholar", "arXiv", "PubMed", "Google Scholar"],
        "features": ["Anti-blocking measures", "Multi-source search", "AI-powered query optimization", "Result deduplication"]
    }
