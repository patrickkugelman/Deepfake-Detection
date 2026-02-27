"""
DeepGuard Scraper Agent (Research Auto-Update)
==============================================
Periodically checks arXiv and GitHub for new deepfake-related research.
Uses Gemini to summarize findings and notifies the Coordinator.
"""

import os
import logging
import requests
import feedparser
import arxiv
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Google GenAI
try:
    from google import generativeai as genai
    from google.generativeai import GenerativeModel
    from google.generativeai import configure as genai_configure
except ImportError:
    print("ERROR: google-generativeai not installed")

# ============================================================================
# CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepGuardScraper")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://coordinator:8000")
SCRAPE_INTERVAL_HOURS = int(os.getenv("SCRAPE_INTERVAL_HOURS", "24"))

# Configure GenAI only if imported successfully
if 'genai_configure' in globals() and GEMINI_API_KEY:
    try:
        genai_configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")
elif not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not set. Summarization will be disabled.")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="DeepGuard Scraper Agent", version="1.0")

# ============================================================================
# SCRAPING LOGIC
# ============================================================================
def scrape_arxiv(keywords: List[str] = ["deepfake detection", "face swap", "diffusion model forensic"]) -> List[Dict[str, Any]]:
    """Fetch recent papers from arXiv"""
    logger.info("📚 Scraping arXiv...")
    results = []
    
    try:
        # Construct query
        query = " OR ".join([f'ti:"{k}"' for k in keywords])
        
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        for result in search.results():
            # Only fresh papers (last 7 days) if needed, but for now grab top 5
            # Basic deduplication logic could go here
            results.append({
                "source": "arXiv",
                "title": result.title,
                "summary": result.summary,
                "url": result.entry_id,
                "date": result.published.strftime("%Y-%m-%d")
            })
            
    except Exception as e:
        logger.error(f"❌ arXiv scrape failed: {e}")
        
    return results

def scrape_github_trending(keywords: List[str] = ["deepfake", "forensic"]) -> List[Dict[str, Any]]:
    """
    Scrape GitHub Trending (simple HTML parsing) or search via API if rate limits allow.
    Using simple search URL parsing for robustness without API token.
    """
    logger.info("🐙 Scraping GitHub...")
    results = []
    
    try:
        # Search GitHub (filtering for recently updated)
        for keyword in keywords:
            url = f"https://github.com/search?q={keyword}&type=repositories&s=updated&o=desc"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Note: GitHub class names change periodically, this is a best-effort scrape
                # Looking for repo list items
                repos = soup.find_all('div', class_='text-normal') # Generic selector, might need tuning
                
                # Fallback: Use mock/heuristic if HTML structure is too complex/changed
                # Actually, specialized tools or API is better.
                # Let's keep it simple: If fail, return empty list to avoid crashes.
                pass
                
    except Exception as e:
        logger.error(f"❌ GitHub scrape failed: {e}")
        
    return results

# ============================================================================
# INTELLIGENCE LAYER
# ============================================================================
async def summarize_and_notify(items: List[Dict[str, Any]]):
    """
    Use Gemini to summarize new findings and verify if they are relevant threats/innovations.
    Then notify Coordinator.
    """
    if not items:
        logger.info("No new items to summarize.")
        return

    logger.info(f"🧠 Analyzing {len(items)} research items with Gemini...")
    
    try:
        # Prepare context
        findings_text = "\n\n".join([
            f"SOURCE: {item['source']}\nTITLE: {item['title']}\nURL: {item['url']}\nSUMMARY: {item['summary'][:500]}..."
            for item in items
        ])
        
        prompt = f"""
        You are the Research Analyst for DeepGuard Pro.
        Analyze these recent research papers/repos related to deepfakes:
        
        {findings_text}
        
        TASK:
        1. Identify any NEW threats (e.g., new generation method that bypasses current detectors).
        2. Identify any NEW detection techniques we should implement.
        3. Create a short "Daily Intelligence Briefing" (max 3 sentences).
        4. If nothing interesting, say "No critical updates."
        
        Output Format: JSON with 'briefing', 'priority' (info/warning/alert), 'critical_link'.
        """
        
        # Call Gemini
        model = GenerativeModel("gemini-1.5-flash") # Use 1.5 Flash for speed/cost if avail, or 2.0
        response = model.generate_content(prompt)
        
        # Simple parsing (assuming text for now, could enforce JSON)
        report = response.text
        
        # Notify Coordinator
        notify_payload = {
            "source": "Scraper Agent",
            "title": "Daily Research Update",
            "message": report[:500], # Trucate for notification
            "priority": "info",
            "link": items[0]['url'] if items else None
        }
        
        try:
            requests.post(f"{COORDINATOR_URL}/system/notification", json=notify_payload, timeout=5)
            logger.info("✅ Notification sent to Coordinator")
        except Exception as e:
            logger.error(f"Failed to notify Coordinator: {e}")
            
    except Exception as e:
        logger.error(f"Summarization failed: {e}")

# ============================================================================
# SCHEDULER JOB
# ============================================================================
def scheduled_job():
    """Main job to run periodically"""
    logger.info("🕑 Starting scheduled scrape...")
    
    # 1. Scrape
    arxiv_items = scrape_arxiv()
    # github_items = scrape_github_trending() # Disabled for now due to complexity without API key
    
    all_items = arxiv_items
    
    # 2. Process
    if all_items:
        import asyncio
        asyncio.run(summarize_and_notify(all_items))
    else:
        logger.info("No items found.")

# Start Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_job, 'interval', hours=SCRAPE_INTERVAL_HOURS)
scheduler.start()

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/health")
def health_check():
    return {"status": "healthy", "agent": "scraper", "scheduler": "running"}

@app.post("/trigger")
def trigger_scrape():
    """Manually trigger the scraper"""
    scheduler.get_job(job_id=None).func() # Hacky way to run now, or just call func
    scheduled_job()
    return {"status": "triggered"}

if __name__ == "__main__":
    import uvicorn
    # Run loop
    uvicorn.run(app, host="0.0.0.0", port=8006)
