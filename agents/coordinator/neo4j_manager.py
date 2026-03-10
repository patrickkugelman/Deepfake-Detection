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
                return [record.data() for record in session.run(query, email=user_email)]
        except Exception as e:
            logger.error(f"⚠️ Failed to fetch history: {e}")
            return []

    def save_fingerprint(self, file_hash: str, model_type: str, confidence: float,
                         dct_concentration: float, fft_high_freq: float,
                         spectral_entropy: float, color_correlation: float):
        """
        Salvează amprenta spectrală (GAN/Diffusion fingerprint) în Knowledge Graph.
        Leagă fingerprint-ul de fișierul analizat.
        """
        if not self.driver:
            logger.warning("⚠️ Neo4j unavailable — fingerprint not saved")
            return

        fp_id = f"fp_{file_hash[:12]}"
        query = """
        MERGE (f:File {hash: $hash})
        MERGE (fp:Fingerprint {id: $fp_id})
        SET fp.model_type            = $model_type,
            fp.confidence            = $confidence,
            fp.dct_energy_concentration = $dct_concentration,
            fp.fft_high_freq_ratio   = $fft_high_freq,
            fp.spectral_entropy      = $spectral_entropy,
            fp.color_correlation     = $color_correlation,
            fp.created_at            = datetime()
        MERGE (f)-[:HAS_FINGERPRINT]->(fp)
        """
        try:
            with self.driver.session() as session:
                session.run(query,
                            hash=file_hash,
                            fp_id=fp_id,
                            model_type=model_type,
                            confidence=confidence,
                            dct_concentration=dct_concentration,
                            fft_high_freq=fft_high_freq,
                            spectral_entropy=spectral_entropy,
                            color_correlation=color_correlation)
            logger.info(f"🔬 Fingerprint saved: {model_type} ({confidence:.1%}) for {file_hash[:8]}")
        except Exception as e:
            logger.error(f"⚠️ Failed to save fingerprint: {e}")

    def get_fingerprint_stats(self) -> dict:
        """
        Returnează statistici despre fingerprint-urile stocate.
        Folosit pentru dashboard și rapoarte.
        """
        if not self.driver:
            return {}

        query = """
        MATCH (fp:Fingerprint)
        RETURN fp.model_type as model, count(fp) as count, avg(fp.confidence) as avg_conf
        ORDER BY count DESC
        """
        try:
            with self.driver.session() as session:
                results = list(session.run(query))
                return {
                    row["model"]: {
                        "count": row["count"],
                        "avg_confidence": round(row["avg_conf"] or 0, 3)
                    }
                    for row in results if row["model"]
                }
        except Exception as e:
            logger.error(f"⚠️ Failed to fetch fingerprint stats: {e}")
            return {}