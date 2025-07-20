from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import urllib.parse
import time
import logging

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
app = FastAPI(title="Academic Paper Search API", version="1.0.0")

# ==== Data Models ====
class PromptRequest(BaseModel):
    query: str

class PaperCard(BaseModel):
    title: str
    authors: str
    snippet: str
    link: str

# ==== Gemini-based Query Formatter ====
def generate_search_query(user_prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Convert the following user intent into a concise Google Scholar academic search query.
        Return only the search query without any additional text or formatting:
        
        User query: {user_prompt}
        
        Academic search query:"""
        
        response = model.generate_content(prompt)
        return response.text.strip().strip('"').strip("'")
    except Exception as e:
        logger.error(f"Error generating search query: {e}")
        # Fallback to original query if Gemini fails
        return user_prompt

# ==== Gemini-based Snippet Summarizer ====
def summarize_snippet(snippet: str) -> str:
    try:
        # Skip summarization for very short snippets
        if len(snippet) < 100:
            return snippet
            
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Summarize the following academic paper snippet in one clear, concise sentence.
        Focus on the main finding or contribution. Return only the summary:
        
        Snippet: {snippet}
        
        Summary:"""
        
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        # Fallback to original snippet if summary is too short or failed
        if len(summary) < 20:
            return snippet
        return summary
    except Exception as e:
        logger.error(f"Error summarizing snippet: {e}")
        return snippet  # Return original snippet if summarization fails

# ==== Optional Google Scholar Backup Endpoint ====
@app.post("/search-scholar", response_model=List[PaperCard])
async def search_papers_scholar(data: PromptRequest):
    """Backup endpoint using Google Scholar (may be blocked)"""
    try:
        # Validate input
        if not data.query or not data.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate smart search query
        smart_query = generate_search_query(data.query)
        logger.info(f"Using Google Scholar for query: {smart_query}")
        
        # Properly encode the search query
        encoded_query = urllib.parse.quote_plus(smart_query)
        search_url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en&num=10"
        
        # Headers to mimic a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://scholar.google.com/",
        }

        # Make request with timeout
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Check if we got blocked
        if "blocked" in response.text.lower():
            raise HTTPException(status_code=503, detail="Google Scholar has blocked this request")
        
        results = []
        paper_results = soup.select(".gs_ri")[:10]
        
        if not paper_results:
            return []
        
        for idx, result in enumerate(paper_results):
            try:
                # Extract title
                title_tag = result.select_one(".gs_rt")
                title = title_tag.get_text(strip=True) if title_tag else "No title available"
                title = title.replace("[PDF]", "").replace("[HTML]", "").strip()
                
                # Extract link
                link = extract_clean_link(title_tag)
                
                # Extract snippet
                snippet_tag = result.select_one(".gs_rs")
                raw_snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No snippet available"
                
                # Extract authors
                author_tag = result.select_one(".gs_a")
                authors = author_tag.get_text(strip=True) if author_tag else "No author information"
                
                # Summarize snippet
                summarized_snippet = summarize_snippet(raw_snippet)
                
                paper_card = PaperCard(
                    title=title,
                    authors=authors,
                    snippet=summarized_snippet,
                    link=link
                )
                
                results.append(paper_card)
                
                if idx > 0:
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing Scholar result {idx}: {e}")
                continue
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scholar search error: {e}")
        raise HTTPException(status_code=500, detail="Google Scholar search failed")

# ==== Main Endpoint using Semantic Scholar API ====
@app.post("/search", response_model=List[PaperCard])
async def search_papers(data: PromptRequest):
    try:
        # Validate input
        if not data.query or not data.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate smart search query
        smart_query = generate_search_query(data.query)
        logger.info(f"Original query: {data.query}")
        logger.info(f"Generated search query: {smart_query}")
        
        # Use Semantic Scholar API
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": smart_query,
            "limit": 10,
            "fields": "title,authors,abstract,url,year,venue,citationCount,externalIds"
        }
        
        headers = {
            "User-Agent": "Academic-Search-API/1.0 (research purposes)"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data_response = response.json()
        papers = data_response.get("data", [])
        
        if not papers:
            logger.warning("No results found from Semantic Scholar")
            return []
        
        results = []
        
        for idx, paper in enumerate(papers):
            try:
                # Extract title
                title = paper.get("title", "No title available")
                
                # Extract authors
                authors_list = paper.get("authors", [])
                if authors_list:
                    author_names = [author.get("name", "Unknown") for author in authors_list]
                    authors_str = ", ".join(author_names[:3])  # Limit to first 3 authors
                    if len(authors_list) > 3:
                        authors_str += " et al."
                else:
                    authors_str = "No author information"
                
                # Add year and venue info
                year = paper.get("year")
                venue = paper.get("venue")
                citation_count = paper.get("citationCount", 0)
                
                if year or venue:
                    additional_info = []
                    if year:
                        additional_info.append(str(year))
                    if venue:
                        additional_info.append(venue)
                    if citation_count > 0:
                        additional_info.append(f"Citations: {citation_count}")
                    
                    authors_str += f" - {', '.join(additional_info)}"
                
                # Extract abstract/snippet
                abstract = paper.get("abstract", "")
                if abstract:
                    # Limit abstract length and summarize if needed
                    if len(abstract) > 300:
                        # Use Gemini to summarize long abstracts
                        snippet = summarize_snippet(abstract)
                    else:
                        snippet = abstract
                else:
                    snippet = "No abstract available"
                
                # Extract URL
                paper_url = paper.get("url", "")
                if not paper_url:
                    # Try to construct URL from external IDs
                    external_ids = paper.get("externalIds", {})
                    if external_ids.get("DOI"):
                        paper_url = f"https://doi.org/{external_ids['DOI']}"
                    elif external_ids.get("ArXiv"):
                        paper_url = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
                    else:
                        paper_url = "No direct link available"
                
                paper_card = PaperCard(
                    title=title,
                    authors=authors_str,
                    snippet=snippet,
                    link=paper_url
                )
                
                results.append(paper_card)
                
                # Add small delay for Gemini API if we're summarizing
                if idx > 0 and len(abstract) > 300:
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing paper {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results)} results from Semantic Scholar")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error connecting to Semantic Scholar: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch data from Semantic Scholar API")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ==== Health Check Endpoint ====
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Academic Paper Search API is running"}

# ==== Root Endpoint ====
@app.get("/")
async def root():
    return {
        "message": "Academic Paper Search API",
        "endpoints": {
            "POST /search": "Search for academic papers",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }