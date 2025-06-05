import json
from collections import defaultdict
from typing import Any, Dict, List

from pyvis.network import Network

from .config import settings


class GraphVisualizer:
    def __init__(self):
        self.node_colors = settings.GRAPH_NODE_COLORS

    def create_3d_graph(
            self,
            entities: List[Dict],
            relationships: List[Dict]) -> str:

        net = Network(
            height="600px",
            width="100%",
            bgcolor="#1a202c",
            font_color="white",
            notebook=False,
            cdn_resources="in_line",
        )

        net.set_options(
            """
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000,
                    "updateInterval": 25
                },
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 150,
                    "springConstant": 0.04,
                    "damping": 0.95,
                    "avoidOverlap": 1
                },
                "minVelocity": 0.75,
                "solver": "barnesHut"
            },
            "nodes": {
                "font": {
                    "size": 14,
                    "color": "white"
                }
            },
            "edges": {
                "font": {
                    "size": 12,
                    "color": "white",
                    "strokeWidth": 0
                },
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                },
                "smooth": {
                    "type": "continuous"
                }
            },
            "interaction": {
                "hover": true,
                "zoomView": true,
                "dragView": true,
                "dragNodes": true,
                "navigationButtons": true,
                "keyboard": {
                    "enabled": true,
                    "speed": {
                        "x": 10,
                        "y": 10,
                        "zoom": 0.02
                    }
                }
            },
            "layout": {
                "improvedLayout": true,
                "hierarchical": {
                    "enabled": false
                }
            }
        }
        """
        )

        node_ids = set()
        for entity in entities:
            color = self.node_colors.get(entity["type"], "#95A5A6")
            size = 20 + (entity.get("count", 1) * 2)

            net.add_node(
                entity["id"],
                label=entity["label"],
                title=f"{entity['label']}\nType: {entity['type']}\nConfidence: {entity['confidence']:.2f}",
                color=color,
                size=size,
                font={"size": 14},
                physics=True,
                mass=2,
            )
            node_ids.add(entity["id"])

        edge_colors = {
            "MITIGATES": "#27AE60",
            "CAUSES": "#E74C3C",
            "AFFECTS": "#F39C12",
            "OWNS": "#3498DB",
            "REQUIRES": "#9B59B6",
            "IMPLEMENTS": "#1ABC9C",
            "MONITORS": "#34495E",
            "COMPLIES_WITH": "#16A085",
        }

        skipped_edges = 0
        for rel in relationships:
            if rel["source"] not in node_ids:
                print(
                    f"Warning: Source node '{rel['source']}' not found for relationship {rel['type']}"
                )
                skipped_edges += 1
                continue
            if rel["target"] not in node_ids:
                print(
                    f"Warning: Target node '{rel['target']}' not found for relationship {rel['type']}"
                )
                skipped_edges += 1
                continue

            color = edge_colors.get(rel["type"], "#7F8C8D")
            net.add_edge(
                rel["source"],
                rel["target"],
                title=f"{rel['type']} (confidence: {rel.get('confidence', 1.0):.2f})",
                label=rel["type"],
                color=color,
                width=rel.get(
                    "confidence",
                    1.0) * 3,
            )

        if skipped_edges > 0:
            print(f"Skipped {skipped_edges} edges due to missing nodes")

        html = net.generate_html()

        custom_html = html.replace(
            "</body>",
            """
            <div style="position: absolute; top: 10px; right: 10px; z-index: 1000;">
                <button onclick="togglePhysics()" style="
                    background: #4ECDC4;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                ">Stop/Start Movement</button>
            </div>
            <script>
                var physicsEnabled = true;
                function togglePhysics() {
                    physicsEnabled = !physicsEnabled;
                    network.setOptions({physics: {enabled: physicsEnabled}});
                }
                setTimeout(function() {
                    network.setOptions({physics: {enabled: false}});
                    physicsEnabled = false;
                }, 5000);
            </script>
            </body>
            """,
        )

        return custom_html

    def get_top_entities(
        self, entities: List[Dict], relationships: List[Dict], n: int = 10
    ) -> List[Dict]:

        connection_counts = defaultdict(int)

        for rel in relationships:
            connection_counts[rel["source"]] += 1
            connection_counts[rel["target"]] += 1

        entity_lookup = {e["id"]: e for e in entities}

        top_entities = []
        for entity_id, count in sorted(
            connection_counts.items(), key=lambda x: x[1], reverse=True
        )[:n]:
            if entity_id in entity_lookup:
                entity = entity_lookup[entity_id].copy()
                entity["connections"] = count
                top_entities.append(entity)

        return top_entities

    def generate_graph_stats(
        self, entities: List[Dict], relationships: List[Dict]
    ) -> Dict[str, Any]:

        type_counts = defaultdict(int)
        for entity in entities:
            type_counts[entity["type"]] += 1

        rel_type_counts = defaultdict(int)
        for rel in relationships:
            rel_type_counts[rel["type"]] += 1

        max_possible_edges = len(entities) * (len(entities) - 1)
        density = (
            len(relationships) /
            max_possible_edges if max_possible_edges > 0 else 0)

        connected_nodes = set()
        for rel in relationships:
            connected_nodes.add(rel["source"])
            connected_nodes.add(rel["target"])

        isolated_nodes = [
            e for e in entities if e["id"] not in connected_nodes]

        return {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entity_type_distribution": dict(type_counts),
            "relationship_type_distribution": dict(rel_type_counts),
            "graph_density": density,
            "isolated_nodes": len(isolated_nodes),
            "connected_nodes": len(connected_nodes),
        }

    def export_to_json(
            self,
            entities: List[Dict],
            relationships: List[Dict]) -> str:

        graph_data = {
            "nodes": entities,
            "edges": relationships,
            "metadata": {
                "node_count": len(entities),
                "edge_count": len(relationships),
                "node_types": list(set(e["type"] for e in entities)),
                "edge_types": list(set(r["type"] for r in relationships)),
            },
        }

        return json.dumps(graph_data, indent=2)
