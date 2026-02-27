# 🚀 Deployment Guide - Complete Docker Setup

## Step-by-Step Commands to Deploy the System

### Step 1: Stop and Remove Existing Containers

```powershell
cd D:\deepfake-detector-structure

# Stop all running containers
docker-compose down

# Optional: Remove old images (for clean rebuild)
docker image prune -a
```

### Step 2: Rebuild All Services with Latest Code

```powershell
# Rebuild all services (this includes all your fixes)
docker-compose build --no-cache

# Or rebuild specific services if you only changed certain ones:
docker-compose build --no-cache coordinator vision_agent audio_agent video_agent bio_agent
```

### Step 3: Start All Services

```powershell
# Start all services in detached mode
docker-compose up -d

# Or start and see logs in real-time (press Ctrl+C to stop)
docker-compose up
```

### Step 4: Initialize Neo4j Knowledge Graph

```powershell
# Wait a few seconds for Neo4j to be ready, then initialize
Start-Sleep -Seconds 10

# Initialize knowledge graph with symbolic rules
docker exec nasa_brain python setup_knowledge.py
```

### Step 5: Verify All Services Are Running

```powershell
# Check container status
docker ps

# Check specific service logs
docker logs nasa_brain --tail 30
docker logs nasa_audio_agent --tail 20
docker logs nasa_vision_agent --tail 20
docker logs nasa_bio_agent --tail 20

# Check if Neo4j is accessible
docker exec nasa_neo4j cypher-shell -u neo4j -p nasa_project_secret "RETURN 'Neo4j OK' as status;"
```

### Step 6: Test the System

```powershell
# Test coordinator endpoint
curl http://localhost:8000/docs

# Or open in browser:
# - Coordinator API: http://localhost:8000/docs
# - Frontend UI: http://localhost:8501
# - Neo4j Browser: http://localhost:7474 (user: neo4j, password: nasa_project_secret)
```

## Quick All-in-One Command Sequence

```powershell
cd D:\deepfake-detector-structure

# Complete rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Wait for services to start
Start-Sleep -Seconds 15

# Initialize Neo4j
docker exec nasa_brain python setup_knowledge.py

# Verify
docker ps
```

## Service URLs

- **Coordinator API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474
- **Audio Agent**: http://localhost:8004
- **Vision Agent**: http://localhost:8002
- **Video Agent**: http://localhost:8003
- **Text Agent**: http://localhost:8001
- **Bio Agent**: http://localhost:8005

## Troubleshooting Commands

```powershell
# View logs for all services
docker-compose logs -f

# View logs for specific service
docker-compose logs -f coordinator
docker-compose logs -f audio_agent

# Restart a specific service
docker-compose restart coordinator

# Rebuild and restart a specific service
docker-compose up -d --build coordinator

# Check service health
docker-compose ps

# Remove everything and start fresh
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## Environment Variables

The coordinator uses:
- `OLLAMA_HOST=http://host.docker.internal:11434` (for Llama 3)
- `NEO4J_URI=bolt://neo4j:7687` (for knowledge graph)

Make sure Ollama is running on your host machine if you want LLM features.

