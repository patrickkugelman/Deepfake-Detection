"""
Conversation Memory System
Stores and retrieves conversation history using Neo4j Knowledge Graph
"""
import logging
from typing import List, Optional, Dict, Any
from neo4j import GraphDatabase
from datetime import datetime
import uuid

logger = logging.getLogger("Memory")

class ConversationMemory:
    """
    Manages conversation persistence and retrieval using Neo4j
    """
    
    def __init__(self, neo4j_uri: str = "bolt://neo4j:7687", 
                 neo4j_auth: tuple = ("neo4j", "nasa_project_secret")):
        self.uri = neo4j_uri
        self.auth = neo4j_auth
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
            self.driver.verify_connectivity()
            logger.info("✅ Memory system connected to Neo4j")
        except Exception as e:
            logger.warning(f"⚠️ Neo4j connection failed for memory: {e} - Memory disabled")
            self.driver = None
    
    def create_conversation(self, user_id: str = "default") -> str:
        """
        Create a new conversation and return conversation_id
        """
        if not self.driver:
            return str(uuid.uuid4())  # Return UUID even if Neo4j unavailable
        
        conversation_id = str(uuid.uuid4())
        try:
            with self.driver.session() as session:
                session.run("""
                    CREATE (c:Conversation {
                        id: $conversation_id,
                        user_id: $user_id,
                        created_at: $created_at,
                        updated_at: $created_at
                    })
                """, conversation_id=conversation_id, user_id=user_id,
                    created_at=datetime.now().isoformat())
                logger.info(f"✅ Created conversation: {conversation_id}")
        except Exception as e:
            logger.error(f"❌ Failed to create conversation: {e}")
        
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   case_id: Optional[str] = None) -> bool:
        """
        Add a message to a conversation
        role: 'user' or 'system'
        """
        if not self.driver:
            return False
        
        try:
            with self.driver.session() as session:
                # Create message node
                message_id = str(uuid.uuid4())
                session.run("""
                    MATCH (c:Conversation {id: $conversation_id})
                    CREATE (m:Message {
                        id: $message_id,
                        role: $role,
                        content: $content,
                        timestamp: $timestamp
                    })
                    CREATE (c)-[:HAS_MESSAGE]->(m)
                    SET c.updated_at = $timestamp
                """, conversation_id=conversation_id, message_id=message_id,
                    role=role, content=content, timestamp=datetime.now().isoformat())
                
                # Link to case if provided
                if case_id:
                    session.run("""
                        MATCH (m:Message {id: $message_id})
                        MATCH (case:Case {id: $case_id})
                        CREATE (m)-[:REFERENCES_CASE]->(case)
                    """, message_id=message_id, case_id=case_id)
                
                logger.debug(f"✅ Added {role} message to conversation {conversation_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to add message: {e}")
            return False
    
    def get_conversation_history(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history as list of messages
        Returns: [{"role": "user", "content": "...", "timestamp": "..."}, ...]
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Conversation {id: $conversation_id})-[:HAS_MESSAGE]->(m:Message)
                    RETURN m.role as role, m.content as content, m.timestamp as timestamp
                    ORDER BY m.timestamp ASC
                    LIMIT $limit
                """, conversation_id=conversation_id, limit=limit)
                
                messages = []
                for record in result:
                    messages.append({
                        "role": record["role"],
                        "content": record["content"],
                        "timestamp": record["timestamp"]
                    })
                
                return messages
        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            return []
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full conversation with metadata
        """
        if not self.driver:
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Conversation {id: $conversation_id})
                    OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message)
                    RETURN c, collect(m) as messages
                """, conversation_id=conversation_id)
                
                record = result.single()
                if not record:
                    return None
                
                conversation = {
                    "id": conversation_id,
                    "user_id": record["c"].get("user_id"),
                    "created_at": record["c"].get("created_at"),
                    "updated_at": record["c"].get("updated_at"),
                    "messages": []
                }
                
                for msg in record["messages"]:
                    if msg:
                        conversation["messages"].append({
                            "role": msg.get("role"),
                            "content": msg.get("content"),
                            "timestamp": msg.get("timestamp")
                        })
                
                return conversation
        except Exception as e:
            logger.error(f"❌ Failed to get conversation: {e}")
            return None
    
    def list_conversations(self, user_id: str = "default", limit: int = 20) -> List[Dict[str, Any]]:
        """
        List conversations for a user
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Conversation {user_id: $user_id})
                    OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message)
                    WITH c, count(m) as message_count
                    RETURN c.id as id, c.created_at as created_at, 
                           c.updated_at as updated_at, message_count
                    ORDER BY c.updated_at DESC
                    LIMIT $limit
                """, user_id=user_id, limit=limit)
                
                conversations = []
                for record in result:
                    conversations.append({
                        "id": record["id"],
                        "created_at": record["created_at"],
                        "updated_at": record["updated_at"],
                        "message_count": record["message_count"]
                    })
                
                return conversations
        except Exception as e:
            logger.error(f"❌ Failed to list conversations: {e}")
            return []
    
    def search_conversations(self, query: str, user_id: str = "default", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search conversations by content (simple text search)
        TODO: Implement vector search with embeddings for semantic search
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Conversation {user_id: $user_id})-[:HAS_MESSAGE]->(m:Message)
                    WHERE m.content CONTAINS $search_term
                    WITH DISTINCT c, count(m) as match_count
                    RETURN c.id as id, c.created_at as created_at, 
                           c.updated_at as updated_at, match_count
                    ORDER BY match_count DESC, c.updated_at DESC
                    LIMIT $limit
                """, user_id=user_id, search_term=query, limit=limit)
                
                conversations = []
                for record in result:
                    conversations.append({
                        "id": record["id"],
                        "created_at": record["created_at"],
                        "updated_at": record["updated_at"],
                        "match_count": record["match_count"]
                    })
                
                return conversations
        except Exception as e:
            logger.error(f"❌ Failed to search conversations: {e}")
            return []
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

# Global memory instance
conversation_memory = ConversationMemory()

