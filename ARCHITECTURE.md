# 🧠 Multimodal AI Forensic System - Architecture

## Three Pillars Implementation

### 🧩 PILLAR 1: Agentic MCP Coordinator (The Brain)

**Location:** `agents/coordinator/`

**Components:**
- `mcp_protocol.py` - Model Context Protocol implementation for agent auto-registration
- `api.py` - Enhanced coordinator with MCP integration and dynamic routing

**Features:**
- ✅ LLM-based cognitive routing (Llama 3)
- ✅ MCP tool registry for agent auto-discovery
- ✅ Dynamic routing based on intermediate results
- ✅ User intent parsing (e.g., "only check audio", "skip video")
- ✅ Modality-aware routing (skips audio agent if video has no audio track)

**MCP Tools Registered:**
- `audio` - Spectral analysis, voice artifacts
- `vision` - Image compression artifacts, facial inconsistencies
- `video` - Temporal inconsistencies, frame-level analysis
- `text` - NLP-based authenticity checks
- `bio` - Physiological signals (pulse, micro-movements)

### 🧠 PILLAR 2: Neuro-Symbolic Debater (The Logic)

**Location:** `agents/coordinator/neuro_symbolic.py`

**Features:**
- ✅ Contradiction detection between neural agents
- ✅ Symbolic rule validation via Neo4j Knowledge Graph
- ✅ Biological impossibility detection (e.g., human face + no pulse = FAKE)
- ✅ Override mechanism: Symbolic logic > Neural predictions
- ✅ Case storage in knowledge graph for learning

**Symbolic Rules:**
1. **CRITICAL**: Human beings MUST have pulse
2. **HIGH**: Talking face REQUIRES audio track
3. **HIGH**: High-confidence contradictions trigger additional validation
4. **CRITICAL**: Human face + No pulse = Biological impossibility → Override to FAKE

**Knowledge Graph Schema:**
- Entities: HumanBeing, Deepfake
- Modalities: Video, Audio, Image, Text
- BioSignals: HeartRate, MicroMovement
- Relationships: MUST_HAVE, REQUIRES, INCONSISTENT_WITH, OVERRIDES

### 🐳 PILLAR 3: Dockerized Microservices

**Services:**
- `coordinator` - MCP orchestrator + Neuro-Symbolic debater
- `audio_agent` - Wav2Vec2 + heuristics
- `vision_agent` - Xception + heuristics
- `video_agent` - Frame analysis + heuristics
- `text_agent` - NLP analysis
- `bio_agent` - Pulse detection, micro-movements (NEW)
- `neo4j` - Knowledge graph database
- `ui` - Frontend interface

## Workflow

1. **Cognitive Routing** → LLM decides which agents to deploy
2. **Phase 1 Execution** → Initial agents analyze
3. **Dynamic Routing** → LLM reviews intermediate results, deploys additional agents if needed
4. **Neuro-Symbolic Validation** → Contradiction detection + symbolic rule checking
5. **Final Verdict** → Symbolic override OR neural consensus
6. **Knowledge Graph Storage** → Case stored for future learning

## Key Innovations

1. **MCP Protocol** - Agents auto-register as tools, enabling dynamic discovery
2. **Two-Phase Routing** - Initial deployment + dynamic expansion based on results
3. **Symbolic Override** - Biological/logical rules override neural predictions
4. **Contradiction Resolution** - Automatic detection and resolution of agent conflicts

## Usage

```bash
# Start all services
docker-compose up -d

# Initialize Neo4j knowledge graph
docker exec nasa_brain python setup_knowledge.py

# Check logs
docker logs nasa_brain -f
```

## API Response Format

```json
{
  "chat_response": "Human-readable analysis",
  "final_verdict": "FAKE" | "REAL",
  "results": {
    "audio": {...},
    "vision": {...}
  },
  "agents_used": ["audio", "vision"],
  "routing_reasoning": "Why these agents were chosen",
  "symbolic_validation": {
    "has_contradiction": false,
    "contradictions": [],
    "symbolic_overrides": [],
    "reasoning": "Neural agents agree"
  },
  "case_id": "uuid"
}
```

