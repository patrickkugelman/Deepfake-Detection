from neo4j import GraphDatabase
import time

# Conectare la containerul Neo4j (use env vars to match docker-compose)
import os
URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "deepguard_secret"))

def create_rules():
    print("🧠 Injecting Logic into Neo4j...")
    
    # Încercăm să ne conectăm (retry logic pentru Docker startup)
    driver = None
    for i in range(10):
        try:
            driver = GraphDatabase.driver(URI, auth=AUTH)
            driver.verify_connectivity()
            break
        except:
            time.sleep(2)
    
    if not driver:
        print("❌ Could not connect to Neo4j Logic Core.")
        return

    query_rules = """
    // Enhanced Knowledge Graph Schema for Neuro-Symbolic Reasoning
    
    // 1. Entities
    CREATE (human:Entity {name: 'HumanBeing', type: 'biological'});
    CREATE (fake:Entity {name: 'Deepfake', type: 'synthetic'});
    
    // 2. Modalities
    CREATE (video:Modality {name: 'Video', type: 'temporal'});
    CREATE (audio:Modality {name: 'Audio', type: 'spectral'});
    CREATE (image:Modality {name: 'Image', type: 'spatial'});
    CREATE (text:Modality {name: 'Text', type: 'linguistic'});
    
    // 3. Bio Signals
    CREATE (pulse:BioSignal {name: 'HeartRate', required: true});
    CREATE (micro_movement:BioSignal {name: 'MicroMovement', required: true});
    
    // 4. CRITICAL RULE: Human beings MUST have pulse
    CREATE (human)-[:MUST_HAVE {severity: 'CRITICAL', rule: 'biological_law'}]->(pulse);
    CREATE (human)-[:MUST_HAVE {severity: 'HIGH', rule: 'biological_law'}]->(micro_movement);
    
    // 5. SYNCHRONIZATION RULES
    CREATE (talking_face:VisualFeature {name: 'TalkingFace'});
    CREATE (voice:AudioFeature {name: 'Voice'});
    CREATE (talking_face)-[:REQUIRES {severity: 'HIGH', rule: 'synchronization'}]->(voice);
    
    // 6. CONTRADICTION RULES
    CREATE (human_face:VisualFeature {name: 'HumanFace'});
    CREATE (human_face)-[:INCONSISTENT_WITH {severity: 'CRITICAL'}]->(fake);
    
    // 7. VALIDATION RULES
    CREATE (high_confidence_fake:NeuralPrediction {threshold: 80.0});
    CREATE (high_confidence_real:NeuralPrediction {threshold: 80.0});
    CREATE (high_confidence_fake)-[:OVERRIDES]->(high_confidence_real);
    """

    with driver.session() as session:
        # Reset database
        print("🔄 Resetting knowledge graph...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create entities
        print("📦 Creating entities...")
        session.run("CREATE (human:Entity {name: 'HumanBeing', type: 'biological'})")
        session.run("CREATE (fake:Entity {name: 'Deepfake', type: 'synthetic'})")
        
        # Create modalities
        print("📹 Creating modalities...")
        session.run("CREATE (video:Modality {name: 'Video', type: 'temporal'})")
        session.run("CREATE (audio:Modality {name: 'Audio', type: 'spectral'})")
        session.run("CREATE (image:Modality {name: 'Image', type: 'spatial'})")
        session.run("CREATE (text:Modality {name: 'Text', type: 'linguistic'})")
        
        # Create bio signals
        print("💓 Creating bio signals...")
        session.run("CREATE (pulse:BioSignal {name: 'HeartRate', required: true})")
        session.run("CREATE (micro_movement:BioSignal {name: 'MicroMovement', required: true})")
        
        # CRITICAL RULE: Human beings MUST have pulse
        print("🔗 Creating biological rules...")
        session.run("""
            MATCH (human:Entity {name: 'HumanBeing'})
            MATCH (pulse:BioSignal {name: 'HeartRate'})
            CREATE (human)-[:MUST_HAVE {severity: 'CRITICAL', rule: 'biological_law'}]->(pulse)
        """)
        session.run("""
            MATCH (human:Entity {name: 'HumanBeing'})
            MATCH (micro:BioSignal {name: 'MicroMovement'})
            CREATE (human)-[:MUST_HAVE {severity: 'HIGH', rule: 'biological_law'}]->(micro)
        """)
        
        # Synchronization rules
        print("🔗 Creating synchronization rules...")
        session.run("CREATE (talking_face:VisualFeature {name: 'TalkingFace'})")
        session.run("CREATE (voice:AudioFeature {name: 'Voice'})")
        session.run("""
            MATCH (tf:VisualFeature {name: 'TalkingFace'})
            MATCH (v:AudioFeature {name: 'Voice'})
            CREATE (tf)-[:REQUIRES {severity: 'HIGH', rule: 'synchronization'}]->(v)
        """)
        
        # Contradiction rules
        print("🔗 Creating contradiction rules...")
        session.run("CREATE (human_face:VisualFeature {name: 'HumanFace'})")
        session.run("""
            MATCH (hf:VisualFeature {name: 'HumanFace'})
            MATCH (fake:Entity {name: 'Deepfake'})
            CREATE (hf)-[:INCONSISTENT_WITH {severity: 'CRITICAL'}]->(fake)
        """)
        
        print("✅ Knowledge Graph Rules Injected successfully!")
    
    driver.close()

if __name__ == "__main__":
    create_rules()