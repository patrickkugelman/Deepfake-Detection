"""
DeepGuard Pro MCP Tools
Defines the tools exposed via MCP Protocol
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class AnalysisMode:
    """Analysis modes for deepfake detection"""
    QUICK = "quick"          # Fast scan (single agent)
    THOROUGH = "thorough"    # Multi-agent analysis
    FORENSIC = "forensic"    # Full analysis + detailed report


class DeepGuardTool(BaseModel):
    """Base tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


# ============================================================================
# TOOL 1: Analyze Media for Deepfakes
# ============================================================================
ANALYZE_DEEPFAKE_TOOL = DeepGuardTool(
    name="analyze_deepfake",
    description="""
    Analyze media files (images, videos, audio) for deepfake manipulation.
    
    This tool uses a multi-agent neural system to detect:
    - Face-swapping (Vision Agent)
    - Voice cloning (Audio Agent)
    - Temporal inconsistencies (Video Agent)
    - Liveness/pulse detection (Bio Agent)
    - AI-generated text (Text Agent)
    
    The system uses Gemini AI for cognitive routing and Neo4j for
    neuro-symbolic reasoning.
    
    Returns a comprehensive analysis report with confidence scores.
    """,
    inputSchema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the media file to analyze (local or URL)"
            },
            "analysis_mode": {
                "type": "string",
                "enum": ["quick", "thorough", "forensic"],
                "default": "thorough",
                "description": "Analysis depth: quick (fast), thorough (recommended), forensic (maximum detail)"
            },
            "file_type": {
                "type": "string",
                "enum": ["image", "video", "audio", "text", "auto"],
                "default": "auto",
                "description": "File type hint (auto-detected if not specified)"
            }
        },
        "required": ["file_path"]
    }
)


# ============================================================================
# TOOL 2: Get System Status
# ============================================================================
SYSTEM_STATUS_TOOL = DeepGuardTool(
    name="get_system_status",
    description="""
    Get the current status of DeepGuard Pro system.
    
    Returns information about:
    - Available agents (Vision, Audio, Video, Bio, Text)
    - System health
    - Neo4j connection status
    - Recent analysis statistics
    """,
    inputSchema={
        "type": "object",
        "properties": {},
        "required": []
    }
)


# ============================================================================
# TOOL 3: Query Analysis History
# ============================================================================
QUERY_HISTORY_TOOL = DeepGuardTool(
    name="query_analysis_history",
    description="""
    Query the Neo4j knowledge graph for previous analysis results.
    
    Useful for:
    - Finding similar analyzed files
    - Checking if a file hash was previously analyzed
    - Retrieving detailed reports from past analyses
    """,
    inputSchema={
        "type": "object",
        "properties": {
            "file_hash": {
                "type": "string",
                "description": "SHA256 hash of the file (optional)"
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "Maximum number of results to return"
            },
            "verdict_filter": {
                "type": "string",
                "enum": ["real", "fake", "uncertain", "all"],
                "default": "all",
                "description": "Filter by verdict type"
            }
        },
        "required": []
    }
)


# ============================================================================
# All Tools Registry
# ============================================================================
ALL_TOOLS: List[DeepGuardTool] = [
    ANALYZE_DEEPFAKE_TOOL,
    SYSTEM_STATUS_TOOL,
    QUERY_HISTORY_TOOL
]


def get_tool_by_name(name: str) -> DeepGuardTool | None:
    """Get tool definition by name"""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None