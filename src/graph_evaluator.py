import json
from collections import defaultdict
from typing import Any, Dict, List, Set

import pandas as pd


class GraphEvaluator:
    def __init__(self):
        self.metrics = {}

    def create_ground_truth_dataset(
        self, document_text: str, expert_annotations: Dict
    ) -> Dict:
        return expert_annotations

    def evaluate_extraction(
        self,
        predicted_entities: List[Dict],
        predicted_relationships: List[Dict],
        ground_truth: Dict,
    ) -> Dict[str, Any]:
        entity_eval = self._evaluate_entities(
            predicted_entities, ground_truth["entities"]
        )

        relationship_eval = self._evaluate_relationships(
            predicted_relationships, ground_truth["relationships"]
        )

        graph_eval = self._evaluate_graph_structure(
            predicted_entities, predicted_relationships
        )

        quality_eval = self._evaluate_quality_indicators(
            predicted_entities, predicted_relationships
        )

        return {
            "entity_metrics": entity_eval,
            "relationship_metrics": relationship_eval,
            "graph_metrics": graph_eval,
            "quality_indicators": quality_eval,
            "overall_score": self._calculate_overall_score(
                entity_eval, relationship_eval
            ),
        }

    def _evaluate_entities(
        self, predicted: List[Dict], ground_truth: List[Dict]
    ) -> Dict:
        pred_set = {(e["label"].lower(), e["type"]) for e in predicted}
        true_set = {(e["label"].lower(), e["type"]) for e in ground_truth}

        true_positives = len(pred_set & true_set)
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        type_metrics = {}
        for entity_type in ["RISK", "CONTROL", "ASSET", "STAKEHOLDER"]:
            type_pred = {
                e["label"].lower() for e in predicted if e["type"] == entity_type
            }
            type_true = {
                e["label"].lower() for e in ground_truth if e["type"] == entity_type
            }

            tp = len(type_pred & type_true)
            fp = len(type_pred - type_true)
            fn = len(type_true - type_pred)

            type_metrics[entity_type] = {
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "support": len(type_true),
            }

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "type_specific": type_metrics,
        }

    def _evaluate_relationships(
        self, predicted: List[Dict], ground_truth: List[Dict]
    ) -> Dict:
        pred_set = {(r["source"].lower(), r["type"], r["target"].lower())
                    for r in predicted}
        true_set = {(r["source"].lower(), r["type"], r["target"].lower())
                    for r in ground_truth}

        true_positives = len(pred_set & true_set)
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    def _evaluate_graph_structure(
        self, entities: List[Dict], relationships: List[Dict]
    ) -> Dict:
        graph = defaultdict(list)
        for rel in relationships:
            graph[rel["source"]].append(rel["target"])

        num_entities = len(entities)
        num_relationships = len(relationships)

        connected_entities = set()
        for rel in relationships:
            connected_entities.add(rel["source"])
            connected_entities.add(rel["target"])

        connectivity_ratio = (
            len(connected_entities) / num_entities if num_entities > 0 else 0
        )

        max_possible_edges = num_entities * (num_entities - 1)
        density = (
            num_relationships /
            max_possible_edges if max_possible_edges > 0 else 0)

        degrees = defaultdict(int)
        for rel in relationships:
            degrees[rel["source"]] += 1
            degrees[rel["target"]] += 1

        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0

        return {
            "num_entities": num_entities,
            "num_relationships": num_relationships,
            "connectivity_ratio": connectivity_ratio,
            "density": density,
            "average_degree": avg_degree,
            "isolated_entities": num_entities - len(connected_entities),
        }

    def _evaluate_quality_indicators(
        self, entities: List[Dict], relationships: List[Dict]
    ) -> Dict:
        entities_with_confidence = [e for e in entities if "confidence" in e]
        high_confidence_entities = [
            e for e in entities if e.get(
                "confidence", 0) > 0.8]
        entities_with_context = [
            e for e in entities if e.get("contexts") or e.get("evidence")
        ]

        relationships_with_confidence = [
            r for r in relationships if "confidence" in r]
        high_confidence_relationships = [
            r for r in relationships if r.get("confidence", 0) > 0.8
        ]

        return {
            "entities_with_confidence": (
                len(entities_with_confidence) /
                len(entities) if entities else 0),
            "high_confidence_entity_ratio": (
                len(high_confidence_entities) /
                len(entities) if entities else 0),
            "entities_with_context_ratio": (
                len(entities_with_context) /
                len(entities) if entities else 0),
            "relationships_with_confidence": (
                len(relationships_with_confidence) /
                len(relationships) if relationships else 0),
            "high_confidence_relationship_ratio": (
                len(high_confidence_relationships) /
                len(relationships) if relationships else 0),
        }

    def _calculate_overall_score(
        self, entity_eval: Dict, relationship_eval: Dict
    ) -> float:
        entity_f1 = entity_eval["f1_score"]
        relationship_f1 = relationship_eval["f1_score"]

        overall = 0.6 * entity_f1 + 0.4 * relationship_f1

        return overall

    def generate_evaluation_report(self, evaluation: Dict) -> str: 
        report = f"""
# Knowledge Graph Extraction Evaluation Report

## Overall Score: {evaluation['overall_score']:.2%}

## Entity Extraction Performance
- Precision: {evaluation['entity_metrics']['precision']:.2%}
- Recall: {evaluation['entity_metrics']['recall']:.2%}
- F1 Score: {evaluation['entity_metrics']['f1_score']:.2%}

### Type-Specific Performance:
"""

        for entity_type, metrics in evaluation["entity_metrics"][
            "type_specific"
        ].items():
            report += f"- {entity_type}: P={metrics['precision']:.2%}, R={metrics['recall']:.2%} (n={metrics['support']})\n"

        report += f"""
## Relationship Extraction Performance
- Precision: {evaluation['relationship_metrics']['precision']:.2%}
- Recall: {evaluation['relationship_metrics']['recall']:.2%}
- F1 Score: {evaluation['relationship_metrics']['f1_score']:.2%}

## Graph Structure
- Total Entities: {evaluation['graph_metrics']['num_entities']}
- Total Relationships: {evaluation['graph_metrics']['num_relationships']}
- Connectivity: {evaluation['graph_metrics']['connectivity_ratio']:.2%}
- Graph Density: {evaluation['graph_metrics']['density']:.4f}

## Quality Indicators
- High Confidence Entities: {evaluation['quality_indicators']['high_confidence_entity_ratio']:.2%}
- Entities with Context: {evaluation['quality_indicators']['entities_with_context_ratio']:.2%}
- High Confidence Relationships: {evaluation['quality_indicators']['high_confidence_relationship_ratio']:.2%}
"""

        return report
