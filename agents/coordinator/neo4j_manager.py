import os
import logging
import time
from typing import Optional, Dict, Any, List
from neo4j import GraphDatabase
from datetime import datetime

logger = logging.getLogger("Cortex.Memory")

class Neo4jManager:
    def __init__(self):
        # Configurații aliniate cu .env și docker-compose.yml
        self.uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "deepguard_secret")
        self.driver = None
        self.connect_with_retry()

    def connect_with_retry(self):
        """Încearcă conectarea la baza de date cu retry logic."""
        max_retries = 10
        for i in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.verify_connectivity()
                logger.info("✅ Connected to Neo4j Knowledge Graph")
                return
            except Exception as e:
                logger.warning(f"⚠️ Neo4j connection attempt {i+1}/{max_retries} failed: {e}")
                time.sleep(3)
        
        logger.error("❌ Failed to connect to Neo4j after all retries. Memory disabled.")
        self.driver = None

    def verify_connectivity(self):
        if self.driver:
            self.driver.verify_connectivity()

    def close(self):
        if self.driver:
            self.driver.close()

    def check_file_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Verifică dacă hash-ul fișierului există deja.
        Returnează verdictul și datele salvate dacă e găsit.
        """
        if not self.driver: return None

        query = """
        MATCH (f:File {hash: $hash})
        OPTIONAL MATCH (f)<-[:UPLOADED]-(u:User)
        RETURN f.verdict as verdict, f.confidence as confidence, 
               f.report as report, f.filename as filename, u.email as uploader
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, hash=file_hash).single()
                if result:
                    logger.info(f"🛡️ MEMORY HIT: File {file_hash[:8]} found!")
                    return {
                        "verdict": result["verdict"],
                        "confidence": result["confidence"],
                        "report": result["report"],
                        "filename": result["filename"],
                        "uploader": result["uploader"],
                        "antivirus_hit": True
                    }
        except Exception as e:
            logger.error(f"⚠️ Neo4j lookup failed: {e}")
            self.connect_with_retry()
        
        return None

    def store_analysis(self, file_hash: str, filename: str, verdict: str, 
                       confidence: float, report: str, agents_used: List[str], 
                       user_email: str = "guest@deepguard.local"):
        """
        Salvează rezultatul analizei și leagă fișierul de un utilizator.
        """
        if not self.driver: 
            logger.error("❌ Cannot store analysis: Neo4j driver not connected")
            return

        query = """
        // 1. Asigură existența utilizatorului
        MERGE (u:User {email: $email})
        ON CREATE SET u.created_at = datetime(), u.role = 'user'
        
        // 2. Crează sau actualizează nodul fișierului
        MERGE (f:File {hash: $hash})
        SET f.filename = $filename,
            f.verdict = $verdict,
            f.confidence = $confidence,
            f.report = $report,
            f.timestamp = datetime()
        
        // 3. Crează relația dintre user și fișier
        MERGE (u)-[r:UPLOADED]->(f)
        SET r.at = datetime()
        
        // 4. Conectează agenții folosiți
        FOREACH (agent_name IN $agents | 
            MERGE (a:Agent {name: agent_name})
            MERGE (f)-[:ANALYZED_BY]->(a)
        )
        """
        try:
            with self.driver.session() as session:
                session.run(query, 
                            email=user_email,
                            hash=file_hash, 
                            filename=filename, 
                            verdict=verdict, 
                            confidence=confidence, 
                            report=report,
                            agents=agents_used)
                logger.info(f"💾 Analysis and User ({user_email}) stored in Knowledge Graph")
        except Exception as e:
            logger.error(f"⚠️ Failed to store in Neo4j: {e}")

    def get_user_history(self, user_email: str):
        """Recuperează istoricul de scanări pentru un utilizator specific."""
        if not self.driver: return []
        
        query = """
        MATCH (u:User {email: $email})-[:UPLOADED]->(f:File)
        RETURN f.filename as name, f.verdict as verdict, f.timestamp as date
        ORDER BY f.timestamp DESC
        """
        try:
            with self.driver.session() as session:
                return list(session.run(query, email=user_email))
        except Exception as e:
            logger.error(f"⚠️ Failed to fetch history: {e}")
            return []