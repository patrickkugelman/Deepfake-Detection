"""
DeepGuard Pro MCP Server
Exposes DeepGuard capabilities via Model Context Protocol
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Sequence
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        LoggingLevel
    )
except ImportError:
    print("ERROR: mcp package not installed. Run: pip install mcp")
    sys.exit(1)

from tools import ALL_TOOLS, get_tool_by_name

# ============================================================================
# Configuration
# ============================================================================
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://coordinator:8000")
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "deepguard-pro")
MCP_SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")
PORT = int(os.getenv("PORT", "3000"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MCP Server Instance
# ============================================================================
mcp_server = Server(MCP_SERVER_NAME)


# ============================================================================
# Tool Handlers
# ============================================================================

@mcp_server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available DeepGuard tools"""
    logger.info("MCP Client requested tool list")
    
    tools = []
    for tool_def in ALL_TOOLS:
        tools.append(Tool(
            name=tool_def.name,
            description=tool_def.description,
            inputSchema=tool_def.inputSchema
        ))
    
    logger.info(f"Returning {len(tools)} tools")
    return tools


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool execution"""
    logger.info(f"Tool call received: {name} with args: {arguments}")
    
    tool = get_tool_by_name(name)
    if not tool:
        error_msg = f"Unknown tool: {name}"
        logger.error(error_msg)
        return [TextContent(type="text", text=f"❌ Error: {error_msg}")]
    
    # Route to appropriate handler
    try:
        if name == "analyze_deepfake":
            return await handle_analyze_deepfake(arguments)
        elif name == "get_system_status":
            return await handle_system_status(arguments)
        elif name == "query_analysis_history":
            return await handle_query_history(arguments)
        else:
            return [TextContent(
                type="text",
                text=f"❌ Tool '{name}' not implemented yet"
            )]
    
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"❌ Error executing {name}: {str(e)}"
        )]


# ============================================================================
# Tool Implementation Handlers
# ============================================================================

async def handle_analyze_deepfake(args: Dict[str, Any]) -> Sequence[TextContent]:
    """Handle deepfake analysis request"""
    file_path = args.get("file_path")
    analysis_mode = args.get("analysis_mode", "thorough")
    file_type = args.get("file_type", "auto")
    
    logger.info(f"Analyzing file: {file_path} (mode: {analysis_mode})")
    
    # Forward to DeepGuard Coordinator
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Prepare request payload
            payload = {
                "file_path": file_path,
                "analysis_mode": analysis_mode,
                "file_type": file_type,
                "source": "mcp_gateway"
            }
            
            # Call coordinator
            response = await client.post(
                f"{COORDINATOR_URL}/chat_analyze",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Format response for MCP
            report = format_analysis_report(result)
            
            logger.info(f"Analysis complete. Verdict: {result.get('verdict')}")
            return [TextContent(type="text", text=report)]
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling coordinator: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ Failed to connect to DeepGuard Coordinator: {str(e)}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ Unexpected error during analysis: {str(e)}"
            )]


async def handle_system_status(args: Dict[str, Any]) -> Sequence[TextContent]:
    """Get DeepGuard system status"""
    logger.info("Fetching system status")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{COORDINATOR_URL}/health")
            response.raise_for_status()
            status = response.json()
            
            report = format_status_report(status)
            return [TextContent(type="text", text=report)]
            
        except Exception as e:
            logger.error(f"Error fetching status: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ Could not fetch system status: {str(e)}"
            )]


async def handle_query_history(args: Dict[str, Any]) -> Sequence[TextContent]:
    """Query analysis history from Neo4j"""
    file_hash = args.get("file_hash")
    limit = args.get("limit", 10)
    verdict_filter = args.get("verdict_filter", "all")
    
    logger.info(f"Querying history (hash: {file_hash}, limit: {limit})")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            params = {
                "limit": limit,
                "verdict": verdict_filter
            }
            if file_hash:
                params["hash"] = file_hash
            
            response = await client.get(
                f"{COORDINATOR_URL}/query_history",
                params=params
            )
            response.raise_for_status()
            history = response.json()
            
            report = format_history_report(history)
            return [TextContent(type="text", text=report)]
            
        except Exception as e:
            logger.error(f"Error querying history: {str(e)}")
            return [TextContent(
                type="text",
                text=f"❌ Could not query analysis history: {str(e)}"
            )]


# ============================================================================
# Report Formatters
# ============================================================================

def format_analysis_report(result: Dict[str, Any]) -> str:
    """Format analysis result for human-readable output"""
    verdict = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    agents_used = result.get("agents_used", [])
    cached = result.get("cached", False)
    
    # Build emoji verdict
    verdict_emoji = {
        "REAL": "✅",
        "FAKE": "🚨",
        "UNCERTAIN": "⚠️",
        "ERROR": "❌"
    }.get(verdict.upper(), "❓")
    
    report = f"""
# 🔍 DeepGuard Pro Analysis Report

## {verdict_emoji} Verdict: **{verdict.upper()}**
**Confidence**: {confidence:.1%}
{'🎯 (Cached Result)' if cached else '🧠 (Fresh Analysis)'}

---

## 📊 Analysis Details

**Agents Deployed**: {', '.join(agents_used) if agents_used else 'None'}

"""
    
    # Add agent-specific findings
    if "agent_results" in result:
        report += "### Agent Findings:\n\n"
        for agent_name, agent_result in result["agent_results"].items():
            agent_verdict = agent_result.get("verdict", "N/A")
            agent_conf = agent_result.get("confidence", 0.0)
            report += f"- **{agent_name.capitalize()}**: {agent_verdict} ({agent_conf:.1%})\n"
    
    # Add executive summary
    if "summary" in result:
        report += f"\n---\n\n## 📝 Executive Summary\n\n{result['summary']}\n"
    
    # Add recommendations
    if "recommendations" in result:
        report += f"\n---\n\n## 💡 Recommendations\n\n{result['recommendations']}\n"
    
    return report.strip()


def format_status_report(status: Dict[str, Any]) -> str:
    """Format system status report"""
    return f"""
# ⚙️ DeepGuard Pro System Status

**Status**: {'🟢 Operational' if status.get('status') == 'healthy' else '🔴 Issues Detected'}

## Available Agents:
{chr(10).join(f"- {agent}" for agent in status.get('agents', []))}

## Database:
- Neo4j: {'✅ Connected' if status.get('neo4j_connected') else '❌ Disconnected'}

## Statistics:
- Total Analyses: {status.get('total_analyses', 'N/A')}
- Cache Hit Rate: {status.get('cache_hit_rate', 'N/A')}
"""


def format_history_report(history: Dict[str, Any]) -> str:
    """Format history query results"""
    records = history.get("records", [])
    
    if not records:
        return "📭 No previous analyses found matching your criteria."
    
    report = f"# 📚 Analysis History\n\nFound **{len(records)}** records:\n\n"
    
    for i, record in enumerate(records, 1):
        report += f"{i}. **{record.get('file_name', 'Unknown')}**\n"
        report += f"   - Verdict: {record.get('verdict')}\n"
        report += f"   - Confidence: {record.get('confidence', 0):.1%}\n"
        report += f"   - Date: {record.get('timestamp', 'N/A')}\n\n"
    
    return report


# ============================================================================
# FastAPI Health Endpoint (for Docker healthcheck)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    logger.info(f"🚀 DeepGuard MCP Gateway starting on port {PORT}")
    logger.info(f"📡 Coordinator URL: {COORDINATOR_URL}")
    yield
    logger.info("🛑 DeepGuard MCP Gateway shutting down")


fastapi_app = FastAPI(
    title="DeepGuard MCP Gateway",
    version=MCP_SERVER_VERSION,
    lifespan=lifespan
)


@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{COORDINATOR_URL}/health")
            coordinator_healthy = response.status_code == 200
    except:
        coordinator_healthy = False
    
    return JSONResponse({
        "status": "healthy" if coordinator_healthy else "degraded",
        "service": "mcp_gateway",
        "version": MCP_SERVER_VERSION,
        "coordinator_reachable": coordinator_healthy
    })


# ============================================================================
# Main Entry Point
# ============================================================================
async def main():
    """Run MCP server via stdio"""
    async with stdio_server() as (read_stream, write_stream):
        logger.info("🎯 MCP Server running on stdio")
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="DeepGuard MCP Gateway")
    parser.add_argument("--stdio", action="store_true", help="Run in stdio mode (for MCP clients)")
    args = parser.parse_args()

    if args.stdio:
        # Run in Stdio mode (for Claude Desktop / Cursor)
        # Note: We don't start uvicorn here to keep stdout clean for JSON-RPC
        asyncio.run(main())
    else:
        # Run in HTTP mode (for Docker container / Health checks)
        logger.info("Starting DeepGuard MCP Gateway (HTTP Mode)...")
        
        config = uvicorn.Config(
            fastapi_app,
            host="0.0.0.0",
            port=PORT,
            log_level="info"
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())