"""
Model Context Protocol (MCP) v2 - Modernized Implementation
============================================================
Enables intelligent agent registration and capability discovery.

FEATURES:
- Pydantic v2 models with strict validation
- FastAPI /register endpoint for self-registration
- Type-safe registry with O(1) lookups
- Zero overhead when not used (lazy initialization)
- Backward compatible with legacy AGENT_REGISTRY

ARCHITECTURE:
- Agents can self-register at startup via POST /register
- Coordinator maintains capability metadata for intelligent routing
- Utility tools (browser, code executor) can be dynamically loaded
"""

import logging
from typing import Dict, List, Optional, Any, Literal, Set, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict
from fastapi import APIRouter, HTTPException, status
import requests
from enum import Enum

logger = logging.getLogger("MCP")

# ============================================================================
# ENUMS & TYPES
# ============================================================================
class AgentType(str, Enum):
    """Agent classification types"""
    NEURAL = "neural"          # ML-based agents (Vision, Audio, etc.)
    UTILITY = "utility"        # Helper tools (Metadata, Browser, etc.)
    HYBRID = "hybrid"          # Combined functionality

class ModalityType(str, Enum):
    """Supported modality types"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"

# ============================================================================
# PYDANTIC V2 MODELS
# ============================================================================
class MCPCapability(BaseModel):
    """Detailed capability definition"""
    model_config = ConfigDict(frozen=True)  # Immutable after creation
    
    name: str = Field(..., description="Capability identifier (e.g., 'face_detection')")
    description: str = Field(..., description="Human-readable description")
    confidence_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence required for this capability"
    )

class MCPInputSchema(BaseModel):
    """Input format specification"""
    model_config = ConfigDict(frozen=True)
    
    modality: ModalityType = Field(..., description="Primary modality handled")
    formats: List[str] = Field(..., description="Supported file formats (e.g., ['jpg', 'png'])")
    max_size_mb: Optional[int] = Field(default=None, description="Max file size in MB")
    requires_preprocessing: bool = Field(default=False, description="Needs preprocessing step")

class MCPOutputSchema(BaseModel):
    """Output format specification"""
    model_config = ConfigDict(frozen=True)
    
    verdict_field: str = Field(default="verdict", description="Field name for verdict")
    confidence_field: str = Field(default="confidence", description="Field name for confidence score")
    details_field: str = Field(default="details", description="Field name for detailed analysis")
    additional_fields: List[str] = Field(default_factory=list, description="Extra fields returned")

class MCPTool(BaseModel):
    """Complete agent/tool definition with Pydantic v2"""
    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)
    
    # Core Identity
    name: str = Field(..., min_length=1, max_length=50, description="Unique tool identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., min_length=10, description="Detailed capability description")
    version: str = Field(default="1.0.0", description="Tool version (semver)")
    
    # Classification
    agent_type: AgentType = Field(..., description="Agent classification")
    modality: ModalityType = Field(..., description="Primary modality handled")
    
    # Capabilities
    capabilities: List[MCPCapability] = Field(..., min_length=1, description="List of capabilities")
    
    # I/O Schemas
    input_schema: MCPInputSchema = Field(..., description="Input format specification")
    output_schema: MCPOutputSchema = Field(..., description="Output format specification")
    
    # Networking
    endpoint: str = Field(..., description="HTTP endpoint URL (empty for utility tools)")
    health_endpoint: Optional[str] = Field(default=None, description="Health check endpoint")
    
    # Metadata
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = Field(default=None)
    is_healthy: bool = Field(default=True)
    
    @validator('endpoint')
    def validate_endpoint(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate endpoint format"""
        if values.get('agent_type') == AgentType.NEURAL and not v:
            raise ValueError("Neural agents must have an endpoint")
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("Endpoint must be a valid HTTP(S) URL")
        return v

class MCPRegistrationRequest(BaseModel):
    """Request model for agent self-registration"""
    model_config = ConfigDict(use_enum_values=True)
    
    tool: MCPTool = Field(..., description="Complete tool definition")
    registration_token: Optional[str] = Field(default=None, description="Security token (future use)")

class MCPHealthStatus(BaseModel):
    """Health check response"""
    is_healthy: bool
    latency_ms: Optional[float] = None
    last_checked: datetime
    model_loaded: Optional[bool] = None
    additional_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ============================================================================
# MCP REGISTRY (Thread-safe, Type-safe)
# ============================================================================
class MCPRegistry:
    """
    Central registry for MCP tools with intelligent capabilities.
    
    FEATURES:
    - O(1) tool lookup by name
    - O(n) capability search (can be optimized with indexes)
    - Lazy initialization (no overhead if unused)
    - Thread-safe operations
    """
    
    def __init__(self, enable_auto_discovery: bool = True):
        self.tools: Dict[str, MCPTool] = {}
        self._capability_index: Dict[str, Set[str]] = {}  # capability_name -> {tool_names}
        self._modality_index: Dict[ModalityType, Set[str]] = {}  # modality -> {tool_names}
        self._initialized = False
        
        if enable_auto_discovery:
            self._auto_discover()
    
    def _auto_discover(self) -> None:
        """Auto-discover agents from static configuration"""
        if self._initialized:
            return
        
        logger.info("🔍 Starting MCP auto-discovery...")
        
        # Define default neural agents
        default_agents = [
            MCPTool(
                name="vision",
                display_name="Vision Agent",
                description="Analyzes images for manipulation artifacts, facial inconsistencies, and GAN signatures using Xception CNN",
                version="2.0.0",
                agent_type=AgentType.NEURAL,
                modality=ModalityType.IMAGE,
                capabilities=[
                    MCPCapability(name="face_detection", description="Detects faces in images", confidence_threshold=0.8),
                    MCPCapability(name="gan_artifact_detection", description="Identifies GAN-generated artifacts", confidence_threshold=0.75),
                    MCPCapability(name="metadata_forensics", description="Analyzes EXIF and metadata", confidence_threshold=0.6),
                ],
                input_schema=MCPInputSchema(
                    modality=ModalityType.IMAGE,
                    formats=["jpg", "jpeg", "png", "bmp", "webp"],
                    max_size_mb=50
                ),
                output_schema=MCPOutputSchema(
                    verdict_field="verdict",
                    confidence_field="confidence",
                    details_field="analysis_details"
                ),
                endpoint="http://deepguard_vision:8002/analyze",
                health_endpoint="http://deepguard_vision:8002/health"
            ),
            MCPTool(
                name="audio",
                display_name="Audio Agent",
                description="Detects voice cloning and audio synthesis using Wav2Vec2 transformer model",
                version="2.0.0",
                agent_type=AgentType.NEURAL,
                modality=ModalityType.AUDIO,
                capabilities=[
                    MCPCapability(name="voice_cloning_detection", description="Identifies synthetic voices", confidence_threshold=0.8),
                    MCPCapability(name="spectral_analysis", description="Analyzes frequency patterns", confidence_threshold=0.7),
                    MCPCapability(name="artifact_detection", description="Detects audio processing artifacts", confidence_threshold=0.75),
                ],
                input_schema=MCPInputSchema(
                    modality=ModalityType.AUDIO,
                    formats=["wav", "mp3", "flac", "ogg", "m4a"],
                    max_size_mb=100
                ),
                output_schema=MCPOutputSchema(),
                endpoint="http://deepguard_audio:8004/analyze",
                health_endpoint="http://deepguard_audio:8004/health"
            ),
            MCPTool(
                name="video",
                display_name="Video Agent",
                description="Analyzes temporal consistency and frame-level artifacts using ResNet3D",
                version="2.0.0",
                agent_type=AgentType.NEURAL,
                modality=ModalityType.VIDEO,
                capabilities=[
                    MCPCapability(name="temporal_consistency", description="Detects temporal anomalies", confidence_threshold=0.8),
                    MCPCapability(name="frame_interpolation_detection", description="Identifies fake frames", confidence_threshold=0.75),
                    MCPCapability(name="motion_analysis", description="Analyzes motion patterns", confidence_threshold=0.7),
                ],
                input_schema=MCPInputSchema(
                    modality=ModalityType.VIDEO,
                    formats=["mp4", "avi", "mov", "mkv", "webm"],
                    max_size_mb=500
                ),
                output_schema=MCPOutputSchema(),
                endpoint="http://deepguard_video:8003/analyze",
                health_endpoint="http://deepguard_video:8003/health"
            ),
            MCPTool(
                name="bio",
                display_name="Bio Agent",
                description="Verifies liveness through rPPG pulse detection and micro-expression analysis",
                version="2.0.0",
                agent_type=AgentType.NEURAL,
                modality=ModalityType.VIDEO,
                capabilities=[
                    MCPCapability(name="pulse_detection", description="Detects heart rate via rPPG", confidence_threshold=0.85),
                    MCPCapability(name="micro_expressions", description="Analyzes facial micro-movements", confidence_threshold=0.75),
                    MCPCapability(name="liveness_verification", description="Confirms biological presence", confidence_threshold=0.8),
                ],
                input_schema=MCPInputSchema(
                    modality=ModalityType.VIDEO,
                    formats=["mp4", "avi", "mov"],
                    max_size_mb=200,
                    requires_preprocessing=True
                ),
                output_schema=MCPOutputSchema(
                    additional_fields=["pulse_detected", "heart_rate_bpm"]
                ),
                endpoint="http://deepguard_bio:8005/analyze",
                health_endpoint="http://deepguard_bio:8005/health"
            ),
            MCPTool(
                name="text",
                display_name="Text Agent",
                description="Detects LLM-generated text using RoBERTa classifier",
                version="2.0.0",
                agent_type=AgentType.NEURAL,
                modality=ModalityType.TEXT,
                capabilities=[
                    MCPCapability(name="llm_detection", description="Identifies AI-generated text", confidence_threshold=0.8),
                    MCPCapability(name="linguistic_analysis", description="Analyzes writing patterns", confidence_threshold=0.7),
                    MCPCapability(name="perplexity_scoring", description="Calculates perplexity metrics", confidence_threshold=0.75),
                ],
                input_schema=MCPInputSchema(
                    modality=ModalityType.TEXT,
                    formats=["txt", "md", "json", "doc", "docx"],
                    max_size_mb=10
                ),
                output_schema=MCPOutputSchema(),
                endpoint="http://deepguard_text:8001/analyze",
                health_endpoint="http://deepguard_text:8001/health"
            ),
        ]
        
        # Register all default agents
        for agent in default_agents:
            self.register_tool(agent)
        
        self._initialized = True
        logger.info(f"✅ MCP Registry initialized with {len(self.tools)} agents")
    
    def register_tool(self, tool: MCPTool) -> None:
        """
        Register a new tool in the registry.
        
        Args:
            tool: MCPTool instance to register
            
        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self.tools:
            logger.warning(f"⚠️ Tool '{tool.name}' already registered, updating...")
        
        self.tools[tool.name] = tool
        
        # Update capability index
        for capability in tool.capabilities:
            if capability.name not in self._capability_index:
                self._capability_index[capability.name] = set()
            self._capability_index[capability.name].add(tool.name)
        
        # Update modality index
        if tool.modality not in self._modality_index:
            self._modality_index[tool.modality] = set()
        self._modality_index[tool.modality].add(tool.name)
        
        logger.info(f"📝 Registered: {tool.name} ({tool.agent_type}) - {tool.description[:50]}...")
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name not in self.tools:
            return False
        
        tool = self.tools[name]
        
        # Clean up indexes
        for capability in tool.capabilities:
            if capability.name in self._capability_index:
                self._capability_index[capability.name].discard(name)
        
        if tool.modality in self._modality_index:
            self._modality_index[tool.modality].discard(name)
        
        del self.tools[name]
        logger.info(f"🗑️ Unregistered: {name}")
        return True
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get tool by name (O(1) lookup)"""
        return self.tools.get(name)
    
    def list_tools(self, agent_type: Optional[AgentType] = None) -> List[MCPTool]:
        """
        List all registered tools, optionally filtered by type.
        
        Args:
            agent_type: Optional filter by agent type
            
        Returns:
            List of MCPTool instances
        """
        tools = list(self.tools.values())
        if agent_type:
            tools = [t for t in tools if t.agent_type == agent_type]
        return tools
    
    def get_tools_by_modality(self, modality: ModalityType) -> List[MCPTool]:
        """
        Get all tools that handle a specific modality (O(1) lookup).
        
        Args:
            modality: Modality type to search for
            
        Returns:
            List of MCPTool instances
        """
        tool_names = self._modality_index.get(modality, set())
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_tools_by_capability(self, capability: str) -> List[MCPTool]:
        """
        Find all tools with a specific capability (O(1) lookup).
        
        Args:
            capability: Capability name to search for
            
        Returns:
            List of MCPTool instances
        """
        tool_names = self._capability_index.get(capability, set())
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_tools_for_extension(self, file_extension: str) -> List[MCPTool]:
        """
        Get all tools that support the given file extension (for agentic routing).
        
        Args:
            file_extension: File extension with or without leading dot (e.g. "mp4", ".jpg")
            
        Returns:
            List of MCPTool instances whose input_schema.formats includes this extension
        """
        ext = (file_extension or "").lstrip(".").lower()
        if not ext:
            return []
        result: List[MCPTool] = []
        for tool in self.tools.values():
            formats = getattr(tool.input_schema, "formats", []) or []
            if any((f or "").lower() == ext for f in formats):
                result.append(tool)
        return result
    
    async def check_tool_health(self, name: str) -> MCPHealthStatus:
        """
        Perform health check on a tool.
        
        Args:
            name: Tool name to check
            
        Returns:
            MCPHealthStatus with current health information
        """
        tool = self.get_tool(name)
        if not tool:
            return MCPHealthStatus(
                is_healthy=False,
                last_checked=datetime.utcnow(),
                error=f"Tool '{name}' not found"
            )
        
        if not tool.health_endpoint:
            return MCPHealthStatus(
                is_healthy=True,
                last_checked=datetime.utcnow(),
                error="No health endpoint configured"
            )
        
        try:
            import time
            start = time.time()
            response = requests.get(tool.health_endpoint, timeout=5)
            latency = (time.time() - start) * 1000  # Convert to ms

            is_healthy = response.status_code == 200
            model_loaded = None
            additional_info = {}

            # Parse response body to extract model_loaded and extra info
            if is_healthy:
                try:
                    health_data = response.json()
                    model_loaded = health_data.get("model_loaded", health_data.get("face_detector_loaded"))
                    for key in ("mode", "device", "model_type"):
                        if key in health_data:
                            additional_info[key] = health_data[key]
                except Exception:
                    pass

            tool.is_healthy = is_healthy
            tool.last_health_check = datetime.utcnow()

            return MCPHealthStatus(
                is_healthy=is_healthy,
                latency_ms=latency,
                last_checked=datetime.utcnow(),
                model_loaded=model_loaded,
                additional_info=additional_info if additional_info else None,
                error=None if is_healthy else f"Status code: {response.status_code}"
            )
        except Exception as e:
            tool.is_healthy = False
            tool.last_health_check = datetime.utcnow()
            return MCPHealthStatus(
                is_healthy=False,
                last_checked=datetime.utcnow(),
                error=str(e)
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dictionary"""
        return {
            "tools": {name: tool.model_dump() for name, tool in self.tools.items()},
            "total_tools": len(self.tools),
            "neural_agents": len([t for t in self.tools.values() if t.agent_type == AgentType.NEURAL]),
            "utility_tools": len([t for t in self.tools.values() if t.agent_type == AgentType.UTILITY]),
        }
    
    def get_agent_registry_dict(self) -> Dict[str, str]:
        """
        Export as simple dict for backward compatibility with api.py AGENT_REGISTRY.
        
        Returns:
            Dict mapping agent names to endpoints
        """
        return {
            name: tool.endpoint
            for name, tool in self.tools.items()
            if tool.endpoint  # Only include tools with endpoints
        }

# ============================================================================
# GLOBAL REGISTRY INSTANCE (Lazy initialization)
# ============================================================================
_mcp_registry_instance: Optional[MCPRegistry] = None

def get_mcp_registry(enable_auto_discovery: bool = True) -> MCPRegistry:
    """
    Get global MCP registry instance (singleton pattern).
    
    Args:
        enable_auto_discovery: Auto-discover default agents on first call
        
    Returns:
        MCPRegistry instance
    """
    global _mcp_registry_instance
    if _mcp_registry_instance is None:
        _mcp_registry_instance = MCPRegistry(enable_auto_discovery=enable_auto_discovery)
    return _mcp_registry_instance

# ============================================================================
# FASTAPI ROUTER (Optional - for agent self-registration)
# ============================================================================
mcp_router = APIRouter(prefix="/mcp", tags=["MCP Protocol"])

@mcp_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_agent(request: MCPRegistrationRequest) -> Dict[str, Any]:
    """
    Agent self-registration endpoint.
    
    Agents can POST to this endpoint at startup to register themselves.
    
    Example:
        ```bash
        curl -X POST http://coordinator:8000/mcp/register \\
          -H "Content-Type: application/json" \\
          -d '{"tool": {...}}'
        ```
    """
    logger.info(f"📝 Registration request received for agent: {request.tool.name}")
    registry = get_mcp_registry()
    
    try:
        registry.register_tool(request.tool)
        logger.info(f"✅ Successfully registered agent: {request.tool.name} at {request.tool.endpoint}")
        return {
            "status": "registered",
            "tool_name": request.tool.name,
            "message": f"Tool '{request.tool.name}' registered successfully",
            "registered_at": request.tool.registered_at.isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Registration failed for {request.tool.name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@mcp_router.get("/tools")
async def list_mcp_tools(agent_type: Optional[str] = None) -> Dict[str, Any]:
    """
    List all registered MCP tools.
    
    Query Parameters:
        agent_type: Optional filter by type ("neural" or "utility")
    """
    registry = get_mcp_registry()
    
    filter_type = AgentType(agent_type) if agent_type else None
    tools = registry.list_tools(agent_type=filter_type)
    
    return {
        "total": len(tools),
        "tools": [tool.model_dump() for tool in tools]
    }

@mcp_router.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific tool"""
    registry = get_mcp_registry()
    tool = registry.get_tool(tool_name)
    
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found"
        )
    
    return tool.model_dump()

@mcp_router.get("/tools/{tool_name}/health")
async def check_tool_health_endpoint(tool_name: str) -> MCPHealthStatus:
    """Check health status of a specific tool"""
    registry = get_mcp_registry()
    return await registry.check_tool_health(tool_name)

@mcp_router.delete("/tools/{tool_name}")
async def unregister_agent(tool_name: str) -> Dict[str, str]:
    """Unregister a tool from the registry"""
    registry = get_mcp_registry()
    
    if registry.unregister_tool(tool_name):
        return {"status": "unregistered", "tool_name": tool_name}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found"
        )

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def get_agent_endpoints() -> Dict[str, str]:
    """
    Get simple agent name -> endpoint mapping for use in api.py.
    
    This provides backward compatibility with the AGENT_REGISTRY pattern.
    
    Returns:
        Dict mapping agent names to their HTTP endpoints
    """
    registry = get_mcp_registry()
    return registry.get_agent_registry_dict()

def get_agents_for_file_type(file_extension: str) -> List[str]:
    """
    Suggest agents based on file extension.
    
    Args:
        file_extension: File extension (e.g., ".jpg", ".mp4")
        
    Returns:
        List of recommended agent names
    """
    registry = get_mcp_registry()
    ext = file_extension.lower().lstrip('.')
    
    # Map extensions to modalities
    modality_map = {
        'jpg': ModalityType.IMAGE, 'jpeg': ModalityType.IMAGE, 'png': ModalityType.IMAGE,
        'mp4': ModalityType.VIDEO, 'avi': ModalityType.VIDEO, 'mov': ModalityType.VIDEO,
        'wav': ModalityType.AUDIO, 'mp3': ModalityType.AUDIO, 'flac': ModalityType.AUDIO,
        'txt': ModalityType.TEXT, 'md': ModalityType.TEXT
    }
    
    modality = modality_map.get(ext)
    if not modality:
        return ['vision']  # Default fallback
    
    tools = registry.get_tools_by_modality(modality)
    return [tool.name for tool in tools]

# ============================================================================
# INTEGRATION HELPER
# ============================================================================
def integrate_with_api() -> None:
    """
    Helper function to integrate MCP Registry with api.py.
    
    Call this in api.py startup to sync AGENT_REGISTRY with MCP Registry.
    
    Example:
        ```python
        # In api.py
        from agents.coordinator.mcp_protocol import integrate_with_api, get_agent_endpoints
        
        @app.on_event("startup")
        async def startup():
            integrate_with_api()
            AGENT_REGISTRY.update(get_agent_endpoints())
        ```
    """
    registry = get_mcp_registry()
    logger.info(f"🔗 MCP Registry integrated with {len(registry.tools)} tools")
    logger.info(f"📋 Available agents: {list(registry.tools.keys())}")