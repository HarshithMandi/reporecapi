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

# ==== Helper function to extract clean link ====
def extract_clean_link(title_tag) -> str:
    if title_tag and title_tag.a:
        href = title_tag.a.get("href", "")
        if href.startswith("/scholar?"):
            # Convert relative URL to absolute
            return f"https://scholar.google.com{href}"
        elif href.startswith("http"):
            return href
    return "No link available"

# ==== Main Endpoint ====
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
        
        # Properly encode the search query
        encoded_query = urllib.parse.quote_plus(smart_query)
        search_url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en&num=10"
        
        # Headers to mimic a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        # Make request with timeout
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Check if we got blocked
        if "blocked" in response.text.lower() or soup.select(".gs_ri") == []:
            logger.warning("Possibly blocked by Google Scholar or no results found")
        
        results = []
        paper_results = soup.select(".gs_ri")[:10]  # Limit to first 10 results
        
        for idx, result in enumerate(paper_results):
            try:
                # Extract title
                title_tag = result.select_one(".gs_rt")
                title = title_tag.get_text(strip=True) if title_tag else "No title available"
                
                # Clean title (remove [PDF], [HTML] etc.)
                title = title.replace("[PDF]", "").replace("[HTML]", "").replace("[CITATION]", "").strip()
                
                # Extract link
                link = extract_clean_link(title_tag)
                
                # Extract snippet
                snippet_tag = result.select_one(".gs_rs")
                raw_snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No snippet available"
                
                # Extract authors and publication info
                author_tag = result.select_one(".gs_a")
                authors = author_tag.get_text(strip=True) if author_tag else "No author information"
                
                # Summarize snippet using Gemini
                summarized_snippet = summarize_snippet(raw_snippet)
                
                # Add small delay to avoid overwhelming Gemini API
                if idx > 0:  # Don't delay on first iteration
                    time.sleep(0.1)
                
                paper_card = PaperCard(
                    title=title,
                    authors=authors,
                    snippet=summarized_snippet,
                    link=link
                )
                
                results.append(paper_card)
                
            except Exception as e:
                logger.error(f"Error processing result {idx}: {e}")
                continue
        
        if not results:
            logger.warning("No results found")
            return []
        
        logger.info(f"Successfully processed {len(results)} results")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch data from Google Scholar")
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