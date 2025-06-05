import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .llm_cache import CachedLLMClient, LLMCache

logger = logging.getLogger(__name__)


class LLMGraphGenerator:
    def __init__(self, provider="openai"):
        self.provider = provider
        self.cache = LLMCache()

        self.primary_model = os.getenv("LLM_MODEL_PRIMARY", "gpt-4o-mini")
        self.fallback_model = os.getenv("LLM_MODEL_FALLBACK", "gpt-3.5-turbo")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

        if provider == "openai":
            try:
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment")

                client = OpenAI(api_key=api_key)
                self.client = CachedLLMClient(client, self.cache)

                self.model = self.primary_model
                self.model_fallback = self.fallback_model

            except ImportError:
                raise ImportError("Please install openai: pip install openai")
            except Exception as e:
                raise Exception(f"Error initializing OpenAI client: {str(e)}")

        elif provider == "anthropic":
            try:
                import anthropic

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Anthropic API key not found in environment")

                client = anthropic.Anthropic(api_key=api_key)
                self.client = CachedLLMClient(client, self.cache)

                self.model = "claude-3-haiku-20240307"
                self.model_fallback = "claude-3-sonnet-20240229"

            except ImportError:
                raise ImportError(
                    "Please install anthropic: pip install anthropic")
            except Exception as e:
                raise Exception(
                    f"Error initializing Anthropic client: {str(e)}")

    def _generate_entity_id(self, label: str, entity_type: str) -> str:
        return hashlib.md5(
            f"{label.lower()}_{entity_type}".encode()).hexdigest()[
            :8]

    def extract_entities_and_relationships(
        self, text: str, use_refinement: bool = False
    ) -> Tuple[List[Dict], List[Dict]]:
        max_text_length = self.max_tokens * 3  # Rough estimate
        text_to_analyze = text[:max_text_length]

        prompt = self._create_extraction_prompt(text_to_analyze)

        try:
            result, from_cache = self._call_llm_with_fallback(prompt)

            if result is None:
                logger.error("Both primary and fallback models failed")
                return [], []

            if not from_cache and "usage" in result:
                logger.info(
                    f"Token usage - Prompt: {result['usage']['prompt_tokens']}, "
                    f"Completion: {result['usage']['completion_tokens']}, "
                    f"Total: {result['usage']['total_tokens']}")

            parsed_result = self._parse_llm_response(result["content"])

            if use_refinement and parsed_result and not from_cache:
                refined_result = self.iterative_refinement(
                    text_to_analyze, parsed_result
                )
                if refined_result:
                    parsed_result = refined_result

            entities, relationships = self._post_process_extraction(
                parsed_result)

            return entities, relationships

        except Exception as e:
            logger.error(f"Error in LLM extraction: {str(e)}")
            import traceback

            traceback.print_exc()
            return [], []

    def _create_extraction_prompt(self, text: str) -> str:
        return f"""
You are an expert in risk assessment and knowledge graph construction.

Analyze the following text and extract:
1. ENTITIES: Identify key entities with their types
2. RELATIONSHIPS: Identify relationships between entities
3. CONFIDENCE: Provide confidence scores based on context clarity
4. REASONING: Explain WHY you identified each entity/relationship

Entity Types:
- RISK: Threats, vulnerabilities, hazards, potential negative events
- CONTROL: Mitigations, safeguards, countermeasures, protective measures
- ASSET: Systems, data, resources being protected, valuable items
- STAKEHOLDER: People, teams, departments, organizations
- IMPACT: Consequences, effects, damages, losses
- COMPLIANCE: Standards, regulations, requirements, frameworks
- THREAT: Threat actors, attackers, sources of harm
- VULNERABILITY: Weaknesses, gaps, deficiencies
- PROCESS: Procedures, workflows, methodologies

Relationship Types:
- MITIGATES: Control reduces/eliminates risk
- AFFECTS: Risk impacts asset
- OWNS: Stakeholder responsible for asset/control
- REQUIRES: Dependencies between entities
- EXPLOITS: Threat exploits vulnerability
- CAUSES: One thing leads to another
- IMPLEMENTS: Stakeholder implements control
- MONITORS: Stakeholder monitors risk/asset

Important: Return ONLY a valid JSON object with this EXACT structure:
{{
    "entities": [
        {{
            "label": "Entity Name",
            "type": "ENTITY_TYPE",
            "confidence": 0.95,
            "reasoning": "Found in context discussing...",
            "evidence": "Quote from text supporting this"
        }}
    ],
    "relationships": [
        {{
            "source_label": "entity_name_1",
            "target_label": "entity_name_2",
            "type": "RELATIONSHIP_TYPE",
            "confidence": 0.85,
            "reasoning": "The text indicates that...",
            "evidence": "Quote showing the relationship"
        }}
    ]
}}

Text to analyze:
{text}
"""

    def _call_llm_with_fallback(
            self, prompt: str) -> Tuple[Optional[Dict], bool]:
        try:
            logger.info(f"Trying primary model: {self.model}")
            result, from_cache = self.client.generate_with_cache(
                prompt, self.model, self.temperature, self.provider
            )
            if result:
                return result, from_cache
        except Exception as e:
            logger.warning(f"Primary model failed: {str(e)}")

        try:
            logger.info(f"Falling back to: {self.model_fallback}")
            result, from_cache = self.client.generate_with_cache(
                prompt, self.model_fallback, self.temperature, self.provider
            )
            return result, from_cache
        except Exception as e:
            logger.error(f"Fallback model also failed: {str(e)}")
            return None, False

    def _parse_llm_response(self, content: str) -> Dict:
        try:
            return json.loads(content)
        except BaseException:
            import re

            json_patterns = [
                r'\{[^{}]*"entities"[^{}]*\}',
                r'\{.*?"entities".*?\}(?=\s*$)',
                r"```json\s*(\{.*?\})\s*```",
                r"```\s*(\{.*?\})\s*```",
                r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})",
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match[0] if isinstance(
                            match, tuple) else match
                        result = json.loads(json_str)
                        if "entities" in result:
                            return result
                    except BaseException:
                        continue

            logger.warning("Could not parse JSON from LLM response")
            return {"entities": [], "relationships": []}

    def _post_process_extraction(
            self, result: Dict) -> Tuple[List[Dict], List[Dict]]:
        processed_entities = []
        entity_map = {}

        for entity in result.get("entities", []):
            if not entity.get("label") or not entity.get("type"):
                continue

            entity_id = self._generate_entity_id(
                entity["label"], entity["type"])
            processed_entity = {
                "id": entity_id,
                "label": entity["label"],
                "type": entity["type"],
                "confidence": float(entity.get("confidence", 0.8)),
                "reasoning": entity.get("reasoning", ""),
                "evidence": entity.get("evidence", ""),
            }
            processed_entities.append(processed_entity)
            entity_map[entity["label"].lower()] = entity_id

        processed_relationships = []
        for rel in result.get("relationships", []):
            source_label = rel.get("source_label", "").lower()
            target_label = rel.get("target_label", "").lower()

            if (
                source_label in entity_map
                and target_label in entity_map
                and rel.get("type")
            ):

                processed_rel = {
                    "source": entity_map[source_label],
                    "target": entity_map[target_label],
                    "type": rel["type"],
                    "confidence": float(rel.get("confidence", 0.7)),
                    "reasoning": rel.get("reasoning", ""),
                    "evidence": rel.get("evidence", ""),
                }
                processed_relationships.append(processed_rel)

        return processed_entities, processed_relationships

    def evaluate_extraction_quality(self,
                                    entities: List[Dict],
                                    relationships: List[Dict],
                                    ground_truth: Dict = None) -> Dict[str,
                                                                       Any]:
        evaluation = {
            "entity_metrics": {
                "total": len(entities),
                "high_confidence": len(
                    [e for e in entities if e.get("confidence", 0) > 0.8]
                ),
                "with_reasoning": len([e for e in entities if e.get("reasoning")]),
                "with_evidence": len([e for e in entities if e.get("evidence")]),
                "avg_confidence": (
                    sum(e.get("confidence", 0) for e in entities) / len(entities)
                    if entities
                    else 0
                ),
            },
            "relationship_metrics": {
                "total": len(relationships),
                "high_confidence": len(
                    [r for r in relationships if r.get("confidence", 0) > 0.8]
                ),
                "with_reasoning": len([r for r in relationships if r.get("reasoning")]),
                "avg_confidence": (
                    sum(r.get("confidence", 0) for r in relationships)
                    / len(relationships)
                    if relationships
                    else 0
                ),
            },
            "type_distribution": {},
        }

        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            evaluation["type_distribution"][entity_type] = (
                evaluation["type_distribution"].get(entity_type, 0) + 1
            )

        if ground_truth:
            true_entities = set(
                e["label"].lower() for e in ground_truth.get("entities", [])
            )
            found_entities = set(e["label"].lower() for e in entities)

            true_positives = len(true_entities & found_entities)
            false_positives = len(found_entities - true_entities)
            false_negatives = len(true_entities - found_entities)

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

            evaluation["ground_truth_comparison"] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }

        return evaluation

    def iterative_refinement(
            self,
            text: str,
            initial_extraction: Dict) -> Dict:
        refinement_prompt = f"""
You are reviewing an entity/relationship extraction from a risk assessment document.

Initial extraction:
{json.dumps(initial_extraction, indent=2)}

Please review and improve this extraction by:
1. Checking for any missed important entities or relationships
2. Verifying the confidence scores are appropriate
3. Correcting any misclassifications
4. Adding any missing context or reasoning
5. Removing any duplicates or incorrect extractions

Consider these specific improvements:
- Are all major risks identified?
- Are controls properly linked to the risks they mitigate?
- Are stakeholder responsibilities clear?
- Is the reasoning for each extraction justified?

Original text excerpt:
{text[:1500]}

Return the refined extraction in the same JSON format, with improvements applied.
Only return the JSON object, no additional text.
"""

        try:
            result, from_cache = self.client.generate_with_cache(
                refinement_prompt,
                self.model_fallback,
                self.temperature,
                self.provider,
            )

            if result:
                refined = self._parse_llm_response(result["content"])
                if refined.get("entities") and len(refined["entities"]) > 0:
                    logger.info("Successfully refined extraction")
                    return refined

        except Exception as e:
            logger.error(f"Error in iterative refinement: {str(e)}")

        return initial_extraction

    def get_cache_stats(self) -> Dict:
        return self.cache.get_stats()

    def clear_expired_cache(self) -> int:
        return self.cache.clear_expired()
