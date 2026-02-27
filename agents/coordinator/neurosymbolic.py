"""
DeepGuard Pro - Neuro-Symbolic Debater
======================================
Combină predicțiile rețelelor neuronale cu reguli simbolice stricte
folosind Neo4j Knowledge Graph pentru a detecta imposibilități biologice.

FIX PRINCIPAL: Confidența se calculează corect pe baza fake_probability.
- Dacă un agent zice REAL cu fake_probability=10%, confidence REAL = 90%
- Dacă un agent zice FAKE cu fake_probability=85%, confidence FAKE = 85%
- Codul vechi folosea fake_probability direct ca confidence, ceea ce făcea
  ca verdictele REAL cu fake_prob mic să aibă confidence mic → INCONCLUSIVE.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any, cast
from neo4j import GraphDatabase
from datetime import datetime

logger = logging.getLogger("DeepGuard.NeuroSymbolic")

# ---------------------------------------------------------------------------
# Agent importance weights for score aggregation
# Higher weight → agent has more influence on the final verdict
# Rationale:
#   - video: ResNet3D temporal analysis is highly specialized for deepfakes → 1.5x
#   - bio: rPPG pulse detection is a strong biological signal → 1.2x
#   - gemini_vision: GPT-4o/Gemini LLM frame-by-frame visual forensics → 1.3x
#   - vision: CNN face analysis, reliable but broad → 1.0x
#   - audio: voice cloning detection, supplementary → 1.0x
#   - text: metadata/text cues, weakest signal → 0.8x
# ---------------------------------------------------------------------------
AGENT_WEIGHTS: Dict[str, float] = {
    "video":         float(os.getenv("WEIGHT_VIDEO_AGENT",       "1.5")),
    "bio":           float(os.getenv("WEIGHT_BIO_AGENT",         "1.2")),
    "gemini_vision": float(os.getenv("WEIGHT_VISION_LLM_AGENT",  "1.3")),
    "vision":        float(os.getenv("WEIGHT_VISION_AGENT",      "1.0")),
    "audio":         float(os.getenv("WEIGHT_AUDIO_AGENT",       "1.0")),
    "text":          float(os.getenv("WEIGHT_TEXT_AGENT",        "0.8")),
}
logger.info(f"Agent weights: {AGENT_WEIGHTS}")


class NeuroSymbolicDebater:
    """
    Nucleul Logic - Validează predicțiile neuronale contra regulilor simbolice.
    """

    def __init__(self,
                 neo4j_uri: Optional[str] = None,
                 neo4j_auth: Optional[Tuple[str, str]] = None):

        self.uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.auth = neo4j_auth or ("neo4j", os.getenv("NEO4J_PASSWORD", "deepguard_secret"))
        self.driver = None
        self._connect()

    def _connect(self):
        """Conectare securizată la Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
            self.driver.verify_connectivity()
            logger.info("✅ Connected to Neo4j Knowledge Graph")
        except Exception as e:
            logger.warning(f"⚠️ Neo4j connection failed: {e} - Symbolic reasoning limited to local rules")
            self.driver = None

    # ==================================================================
    # HELPER: Compute correct confidence from agent result
    # ==================================================================
    @staticmethod
    def _compute_agent_confidence(verdict: str, data: dict) -> float:
        """
        Calculează confidența corectă a unui agent pe baza verdictului și fake_probability.

        Agenții returnează fake_probability (0-100) = probabilitatea ca media să fie FAKE.
        - Vision: verdict=Real, fake_probability=10.75 → confident 89.25% că e REAL
        - Audio:  verdict=Fake, fake_probability=85.0  → confident 85% că e FAKE
        - Text:   verdict=REAL, fake_probability=20.0, confidence=80.0 → confident 80% REAL

        Ordinea de prioritate:
        1. fake_probability (cea mai fiabilă - vine direct din softmax)
        2. confidence (field explicit de la agent, ex: text agent)
        3. default 0.5 (dacă nu avem nimic)
        """
        fake_probability = data.get("fake_probability", None)
        raw_confidence = data.get("confidence", None)

        if fake_probability is not None:
            fp = float(fake_probability)
            # Normalizează la 0-1 dacă e pe scala 0-100
            if fp > 1.0:
                fp = fp / 100.0

            if verdict == "FAKE":
                # fake_probability ESTE confidența pentru verdictul FAKE
                confidence = fp
            elif verdict in ("REAL", "AUTHENTIC"):
                # Pentru REAL, confidența = 1 - fake_probability
                # Ex: fake_prob=10% → 90% confident că e REAL
                confidence = 1.0 - fp
            else:
                confidence = 0.5
        elif raw_confidence is not None:
            # Fallback: folosește câmpul confidence direct (ex: text agent, gemini_vision)
            confidence = float(raw_confidence)
            if confidence > 1.0:
                confidence = confidence / 100.0
            # Safety: 0.0 confidence = agentul nu a putut determina → neutru
            if confidence == 0.0:
                confidence = 0.5
        else:
            confidence = 0.5

        # Clamp la interval valid
        return max(0.0, min(1.0, confidence))

    # ==================================================================
    # MAIN: Validate contradictions & determine final verdict
    # ==================================================================
    def validate_contradictions(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detectează conflicte între agenți și aplică reguli de 'Biological Override'.
        Returnează final_verdict, reasoning, confidence (pentru celery_tasks.py).
        """
        contradictions = []
        symbolic_overrides = []
        neural_verdicts = {}

        # =============================================================
        # 1. Normalize neural results
        # =============================================================
        for agent, result in agent_results.items():
            if not isinstance(result, dict):
                continue

            raw_verdict = str(result.get("verdict", "UNKNOWN")).upper()

            # Skip agenți cu erori sau inconclusive
            if raw_verdict == "ERROR" or raw_verdict == "UNKNOWN":
                logger.warning(f"⚠️ Skipping {agent}: verdict={raw_verdict}")
                continue

            # Skip verdictele INCONCLUSIVE_* (ex: INCONCLUSIVE_TECHNICAL)
            if "INCONCLUSIVE" in raw_verdict:
                logger.warning(f"⚠️ Skipping {agent}: inconclusive verdict ({raw_verdict})")
                continue

            # Normalizează verdictele variate la FAKE/REAL
            if raw_verdict in ("FAKE", "SUSPICIOUS"):
                normalized_verdict = "FAKE"
            elif raw_verdict in ("REAL", "AUTHENTIC"):
                normalized_verdict = "REAL"
            else:
                logger.warning(f"⚠️ Skipping {agent}: unrecognized verdict '{raw_verdict}'")
                continue

            # Calculează confidența corect
            confidence = self._compute_agent_confidence(normalized_verdict, result)

            neural_verdicts[agent] = {
                "verdict": normalized_verdict,
                "confidence": confidence,
                "fake_probability": float(result.get("fake_probability", 0.0))
            }

            logger.info(
                f"📊 {agent}: verdict={normalized_verdict}, "
                f"fake_prob={result.get('fake_probability', 'N/A')}, "
                f"computed_confidence={confidence:.2f}"
            )

        # =============================================================
        # 2. SynthID Ground Truth Override
        # =============================================================
        synthid_res = agent_results.get("synthid", {})
        if isinstance(synthid_res, dict) and synthid_res.get("detected"):
            return {
                "has_contradiction": False,
                "contradictions": [],
                "symbolic_overrides": ["Google SynthID Digital Watermark Detected"],
                "final_verdict": "FAKE",
                "confidence": 0.99,
                "reasoning": "🔴 Google SynthID detected an invisible digital watermark certifying this content as AI-generated.",
                "override_priority": "CRITICAL",
                "neural_consensus": {"fake_agents": [], "real_agents": []}
            }

        # =============================================================
        # 3. Identify Divergences
        # =============================================================
        fake_agents = [a for a, v in neural_verdicts.items() if v["verdict"] == "FAKE"]
        real_agents = [a for a, v in neural_verdicts.items() if v["verdict"] == "REAL"]

        if fake_agents and real_agents:
            contradictions.append(
                f"Neural Conflict: {', '.join(fake_agents)} flagged FAKE, "
                f"while {', '.join(real_agents)} suggested REAL."
            )

        # =============================================================
        # 4. Biological Impossibility Override (Symbolic Rule)
        # =============================================================
        vision_res = agent_results.get("vision", {})
        bio_res = agent_results.get("bio", {})

        if isinstance(bio_res, dict) and isinstance(vision_res, dict):
            bio_status = bio_res.get("status", "OK")
            vision_verdict = str(vision_res.get("verdict", "")).upper()

            is_human_face = (
                "face" in str(vision_res.get("details", "")).lower() or
                vision_verdict in ("REAL", "AUTHENTIC")
            )
            pulse_detected = bio_res.get("pulse_detected", None)

            if bio_status == "OK" and is_human_face and pulse_detected is False:
                reason = "BIOLOGICAL IMPOSSIBILITY: Human face detected but no pulse (rPPG) found."
                symbolic_overrides.append(reason)
                logger.warning(f"🚨 {reason}")

        # =============================================================
        # 5. Determine Final Verdict
        # =============================================================
        has_contradiction = len(contradictions) > 0
        final_verdict = "INCONCLUSIVE"
        override_priority = None
        reasoning = ""
        final_confidence = 0.0

        # --- PRIORITATE 1: Override Simbolic (Biologic) ---
        if symbolic_overrides:
            final_verdict = "FAKE"
            override_priority = "CRITICAL"
            reasoning = " ".join(symbolic_overrides)
            final_confidence = 0.95

        # --- PRIORITATE 2: Weighted Score Aggregation ---
        elif neural_verdicts:
            # Calculează scorul ponderat:
            # FAKE: scor negativ (-1 * confidence)
            # REAL: scor pozitiv (+1 * confidence)
            scores = []
            reasons = []
            weight_total = 0.0

            for agent, data in neural_verdicts.items():
                conf = data["confidence"]
                w = AGENT_WEIGHTS.get(agent, 1.0)
                if data["verdict"] == "FAKE":
                    scores.append(-1.0 * conf * w)
                    reasons.append(f"{agent}(w={w:.1f}) → FAKE ({conf:.0%})")
                elif data["verdict"] == "REAL":
                    scores.append(1.0 * conf * w)
                    reasons.append(f"{agent}(w={w:.1f}) → REAL ({conf:.0%})")
                weight_total += w

            if scores:
                # Weighted average: normalize by total weight so score stays in [-1, +1]
                avg_score = sum(scores) / weight_total if weight_total > 0 else sum(scores) / len(scores)
                # abs(avg_score) = cât de puternic este consensul (0=egal împărțit, 1=toți de acord)
                # Scalăm: consensus_strength in [0,1] → confidence in [0.5, 1.0]
                # astfel un scor mic dar consistent (ex: avg=-0.15) = 57.5% confidence, nu 15%
                consensus_strength = abs(avg_score)
                final_confidence = min(1.0, 0.5 + 0.5 * consensus_strength)

                logger.info(f"🧮 Weighted score aggregation: scores={[f'{s:.2f}' for s in scores]}, total_weight={weight_total:.1f}, avg={avg_score:.3f}")

                # Threshold-uri de decizie
                if avg_score > 0.3:
                    final_verdict = "REAL"
                    reasoning = f"Consens REAL (avg_score={avg_score:.2f}). {'; '.join(reasons)}"
                elif avg_score < -0.3:
                    final_verdict = "FAKE"
                    reasoning = f"Consens FAKE (avg_score={avg_score:.2f}). {'; '.join(reasons)}"
                else:
                    # Zonă ambiguă - folosim votul majoritar
                    fake_count = len(fake_agents)
                    real_count = len(real_agents)

                    if fake_count > real_count:
                        final_verdict = "FAKE"
                        reasoning = f"Majoritate FAKE ({fake_count} vs {real_count}). {'; '.join(reasons)}"
                        # Confidence = media confidențelor agenților FAKE
                        fake_confs = [v["confidence"] for v in neural_verdicts.values() if v["verdict"] == "FAKE"]
                        final_confidence = sum(fake_confs) / len(fake_confs) if fake_confs else 0.5
                    elif real_count > fake_count:
                        final_verdict = "REAL"
                        reasoning = f"Majoritate REAL ({real_count} vs {fake_count}). {'; '.join(reasons)}"
                        real_confs = [v["confidence"] for v in neural_verdicts.values() if v["verdict"] == "REAL"]
                        final_confidence = sum(real_confs) / len(real_confs) if real_confs else 0.5
                    else:
                        # Egalitate perfectă - tie-breaker: agentul cu cea mai mare confidence
                        top_agent_name = max(neural_verdicts, key=lambda a: neural_verdicts[a]["confidence"])
                        top = neural_verdicts[top_agent_name]
                        final_verdict = top["verdict"]
                        final_confidence = top["confidence"]
                        reasoning = f"Tie-break: {top_agent_name} ({top['verdict']} @ {top['confidence']:.0%}). {'; '.join(reasons)}"

                # --- SECURITY OVERRIDE ---
                # Dacă cel puțin un agent e foarte sigur că e FAKE (>80%), respectăm asta
                # chiar dacă media e ambiguă (protecție contra deepfake-urilor subtile)
                if fake_agents and final_verdict != "FAKE":
                    max_fake_conf = max(
                        neural_verdicts[a]["confidence"]
                        for a in fake_agents
                    )
                    if max_fake_conf > 0.80:
                        source_agent = max(fake_agents, key=lambda a: neural_verdicts[a]["confidence"])
                        final_verdict = "FAKE"
                        final_confidence = max_fake_conf
                        reasoning = (
                            f"⚠️ SECURITY OVERRIDE: {source_agent} a detectat FAKE cu "
                            f"confidență {max_fake_conf:.0%}, depășind threshold-ul de securitate. "
                            f"{'; '.join(reasons)}"
                        )
                        logger.warning(f"🚨 Security override triggered by {source_agent} ({max_fake_conf:.0%})")

                if has_contradiction:
                    reasoning += f" | Contradicții: {'; '.join(contradictions)}"
            else:
                final_verdict = "INCONCLUSIVE"
                reasoning = "Nu s-au putut calcula scoruri valide."

        else:
            # Niciun agent valid
            final_verdict = "INCONCLUSIVE"
            reasoning = "Toți agenții au returnat INCONCLUSIVE sau ERROR. Verifică calitatea fișierului."
            final_confidence = 0.0

        logger.info(f"🧠 Final verdict: {final_verdict} (confidence={final_confidence:.2f}) - {reasoning}")

        return {
            "has_contradiction": has_contradiction,
            "contradictions": contradictions,
            "symbolic_overrides": symbolic_overrides,
            "final_verdict": final_verdict,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "override_priority": override_priority,
            "neural_consensus": {
                "fake_agents": fake_agents,
                "real_agents": real_agents
            }
        }

    # ==================================================================
    # STORE: Persist analysis in Neo4j Knowledge Graph
    # ==================================================================
    def store_analysis(self, case_id: str, agent_results: Dict[str, Any], final_verdict: str):
        """Salvează graficul analizei în Neo4j pentru audit ulterior."""
        if not self.driver:
            return

        timestamp = datetime.now().isoformat()

        query = """
        MERGE (c:Case {id: $case_id})
        SET c.verdict = $verdict, c.timestamp = $timestamp
        WITH c
        UNWIND $results AS res
        MERGE (a:Agent {name: res.agent})
        CREATE (r:Result {verdict: res.verdict, confidence: res.prob})
        CREATE (c)-[:HAS_RESULT]->(r)
        CREATE (r)-[:GENERATED_BY]->(a)
        """

        results_list = []
        for agent, res in agent_results.items():
            if isinstance(res, dict) and "error" not in res:
                results_list.append({
                    "agent": agent,
                    "verdict": str(res.get("verdict", "UNKNOWN")).upper(),
                    "prob": float(res.get("fake_probability", 0.0))
                })

        try:
            with self.driver.session() as session:
                session.run(query,
                            case_id=case_id,
                            verdict=final_verdict,
                            timestamp=timestamp,
                            results=results_list)
                logger.info(f"💾 Case {case_id} persisted in Neo4j Knowledge Graph.")
        except Exception as e:
            logger.error(f"❌ Neo4j Storage Error: {e}")

    def close(self):
        if self.driver:
            self.driver.close()
