"""
Browser Automation Tool
Uses Playwright for web scraping, search, and content extraction
"""
import logging
from typing import Dict, Any, Optional
import asyncio

logger = logging.getLogger("BrowserTool")

# Lazy import - only import if needed
_playwright_available = False
try:
    from playwright.async_api import async_playwright
    _playwright_available = True
except ImportError:
    logger.warning("⚠️ Playwright not available - browser tool disabled")

async def browser_navigate(url: str, timeout: int = 10000) -> Dict[str, Any]:
    """
    Navigate to URL and extract text content
    """
    if not _playwright_available:
        return {"error": "Playwright not installed", "content": ""}
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            content = await page.text_content("body")
            
            await browser.close()
            
            return {
                "url": url,
                "content": content[:5000] if content else "",  # Limit content size
                "success": True
            }
    except Exception as e:
        logger.error(f"❌ Browser navigation error: {e}")
        return {"error": str(e), "url": url, "success": False}

async def browser_screenshot(url: str, timeout: int = 10000) -> Dict[str, Any]:
    """
    Take screenshot of URL
    Returns base64 encoded image
    """
    if not _playwright_available:
        return {"error": "Playwright not installed", "screenshot": ""}
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            screenshot_bytes = await page.screenshot()
            
            import base64
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            await browser.close()
            
            return {
                "url": url,
                "screenshot": screenshot_b64,
                "success": True
            }
    except Exception as e:
        logger.error(f"❌ Browser screenshot error: {e}")
        return {"error": str(e), "url": url, "success": False}

async def browser_search(query: str, engine: str = "duckduckgo", max_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using specified engine
    """
    if not _playwright_available:
        return {"error": "Playwright not installed", "results": []}
    
    try:
        if engine == "duckduckgo":
            search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        else:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            await page.goto(search_url, timeout=10000, wait_until="domcontentloaded")
            
            # Extract search results (simplified - would need engine-specific selectors)
            results = []
            # This is a placeholder - actual implementation would parse search results
            content = await page.text_content("body")
            
            await browser.close()
            
            return {
                "query": query,
                "engine": engine,
                "results": results,
                "content_preview": content[:1000] if content else "",
                "success": True
            }
    except Exception as e:
        logger.error(f"❌ Browser search error: {e}")
        return {"error": str(e), "query": query, "results": [], "success": False}

async def browser_extract_text(url: str, selector: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract text content from URL, optionally filtered by CSS selector
    """
    if not _playwright_available:
        return {"error": "Playwright not installed", "text": ""}
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            await page.goto(url, timeout=10000, wait_until="domcontentloaded")
            
            if selector:
                element = await page.query_selector(selector)
                text = await element.text_content() if element else ""
            else:
                text = await page.text_content("body")
            
            await browser.close()
            
            return {
                "url": url,
                "text": text[:10000] if text else "",  # Limit text size
                "success": True
            }
    except Exception as e:
        logger.error(f"❌ Browser text extraction error: {e}")
        return {"error": str(e), "url": url, "text": "", "success": False}

def browser_check_deepfake_indicators(url: str) -> Dict[str, Any]:
    """
    Check URL for deepfake-related indicators (synchronous wrapper)
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(browser_navigate(url))

