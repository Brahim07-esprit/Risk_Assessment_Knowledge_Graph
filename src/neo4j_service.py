import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

from neo4j import GraphDatabase

from .config import settings


class Neo4jService:
    def __init__(self):
        self.driver = None
        self._connect()

    def _connect(self):
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                if self.driver:
                    self.driver.close()

                self.driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60,
                )

                with self.driver.session() as session:
                    session.run("RETURN 1")

                print("✅ Neo4j connection established successfully")
                return

            except Exception as e:
                print(
                    f"❌ Neo4j connection attempt {attempt + 1} failed: {str(e)}")
                if "AuthenticationRateLimit" in str(e):
                    print(
                        f"⏳ Rate limited. Waiting {retry_delay * 2} seconds...")
                    time.sleep(retry_delay * 2)
                elif attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(
                        f"Failed to connect to Neo4j after {max_retries} attempts: {str(e)}"
                    )

    def create_graph(
            self,
            document_name: str,
            entities: List[Dict],
            relationships: List[Dict]) -> str:

        graph_id = str(uuid.uuid4())

        try:
            with self.driver.session() as session:
                session.run(
                    """
                    CREATE (g:Graph {
                        id: $graph_id,
                        document_name: $document_name,
                        created_at: datetime(),
                        entity_count: $entity_count,
                        relationship_count: $relationship_count
                    })
                """,
                    graph_id=graph_id,
                    document_name=document_name,
                    entity_count=len(entities),
                    relationship_count=len(relationships),
                )

                for entity in entities:
                    session.run(
                        """
                        CREATE (e:Entity {
                            id: $id,
                            label: $label,
                            type: $type,
                            confidence: $confidence,
                            graph_id: $graph_id
                        })
                    """,
                        **entity,
                        graph_id=graph_id,
                    )

                for rel in relationships:
                    session.run(
                        """
                        MATCH (a:Entity {id: $source, graph_id: $graph_id})
                        MATCH (b:Entity {id: $target, graph_id: $graph_id})
                        CREATE (a)-[r:RELATIONSHIP {
                            type: $type,
                            confidence: $confidence,
                            graph_id: $graph_id
                        }]->(b)
                    """,
                        graph_id=graph_id,
                        **rel,
                    )

            return graph_id

        except Exception as e:
            if "Unauthorized" in str(e) or "AuthenticationRateLimit" in str(e):
                print(
                    "🔄 Neo4j authentication issue detected, attempting to reconnect..."
                )
                self._connect()
                return self.create_graph(
                    document_name, entities, relationships)
            else:
                raise e

    def get_graph(self, graph_id: str) -> Dict[str, Any]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (g:Graph {id: $graph_id})
                RETURN g
            """,
                graph_id=graph_id,
            )

            graph_data = result.single()
            if not graph_data:
                return None

            entities_result = session.run(
                """
                MATCH (e:Entity {graph_id: $graph_id})
                RETURN e
            """,
                graph_id=graph_id,
            )

            entities = []
            for record in entities_result:
                entity = dict(record["e"])
                entities.append(entity)

            relationships_result = session.run(
                """
                MATCH (a:Entity {graph_id: $graph_id})-[r:RELATIONSHIP]->(b:Entity {graph_id: $graph_id})
                RETURN a.id as source, b.id as target, r.type as type, r.confidence as confidence
            """,
                graph_id=graph_id,
            )

            relationships = []
            for record in relationships_result:
                relationships.append(
                    {
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "confidence": record["confidence"],
                    }
                )

            return {
                "graph_id": graph_id,
                "metadata": dict(graph_data["g"]),
                "entities": entities,
                "relationships": relationships,
            }

    def search_entities(self, query: str, graph_id: str = None) -> List[Dict]:
        with self.driver.session() as session:
            if graph_id:
                result = session.run(
                    """
                    MATCH (e:Entity {graph_id: $graph_id})
                    WHERE toLower(e.label) CONTAINS toLower($query)
                    RETURN e
                    LIMIT 20
                """,
                    graph_id=graph_id,
                    query=query,
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.label) CONTAINS toLower($query)
                    RETURN e
                    LIMIT 20
                """,
                    query=query,
                )

            entities = []
            for record in result:
                entities.append(dict(record["e"]))

            return entities

    def get_entity_relationships(
            self,
            entity_id: str,
            graph_id: str) -> List[Dict]:
        """Get all relationships for a specific entity"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {id: $entity_id, graph_id: $graph_id})-[r]-(other:Entity)
                RETURN e, r, other, type(r) as rel_type
            """,
                entity_id=entity_id,
                graph_id=graph_id,
            )

            relationships = []
            for record in result:
                relationships.append(
                    {
                        "entity": dict(record["e"]),
                        "other": dict(record["other"]),
                        "relationship": dict(record["r"]),
                        "type": record["rel_type"],
                    }
                )

            return relationships

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
