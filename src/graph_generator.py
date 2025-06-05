import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import spacy

from .config import settings

logger = logging.getLogger(__name__)


class GraphGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = int(settings.SPACY_MAX_LENGTH)

        self.risk_patterns = [
            r"\b(?:risk|threat|vulnerability|hazard|danger|exposure)\b",
            r"\b(?:high|medium|low|critical|significant|moderate|minimal)\s+(?:risk|threat|exposure)\b",
            r"\b(?:potential|significant|critical|major|minor|emerging|inherent|residual)\s+(?:risk|threat|vulnerability)\b",
            r"\b(?:operational|financial|strategic|compliance|reputational|cyber|security|safety)\s+(?:risk|threat)\b",
            r"\b(?:data breach|system failure|service disruption|security incident|compliance violation)\b",
        ]

        self.control_patterns = [
            r"\b(?:control|mitigation|safeguard|measure|protection|countermeasure|defense)\b",
            r"\b(?:implement|establish|maintain|monitor|enforce|review)\s+(?:control|measure|safeguard)\b",
            r"\b(?:preventive|detective|corrective|compensating|deterrent)\s+(?:control|measure)\b",
            r"\b(?:access control|encryption|backup|monitoring|audit|training|policy|procedure)\b",
            r"\b(?:ISO 27001|NIST|COBIT|SOC 2|PCI DSS|GDPR compliance)\b",
        ]

        self.asset_patterns = [
            r"\b(?:asset|resource|system|infrastructure|data|information|application)\b",
            r"\b(?:critical|sensitive|valuable|confidential|proprietary)\s+(?:asset|data|system|information)\b",
            r"\b(?:IT|business|operational|customer|financial|intellectual property)\s+(?:asset|system|data)\b",
            r"\b(?:server|database|network|cloud|storage|endpoint|workstation)\b",
            r"\b(?:personal data|PII|PHI|payment card|trade secret|customer record)\b",
        ]

        self.impact_patterns = [
            r"\b(?:impact|consequence|effect|damage|loss|harm|disruption)\b",
            r"\b(?:financial|operational|reputational|legal|regulatory)\s+(?:impact|loss|damage)\b",
            r"\b(?:business interruption|data loss|revenue loss|downtime|breach)\b",
            r"\b(?:cost|penalty|fine|liability|exposure)\b",
        ]

        self.stakeholder_patterns = [
            r"\b(?:stakeholder|owner|manager|department|team|responsible party|custodian)\b",
            r"\b(?:executive|management|IT department|security team|compliance officer|data owner)\b",
            r"\b(?:CISO|CTO|CFO|CEO|DPO|risk manager|auditor)\b",
            r"\b(?:vendor|supplier|partner|customer|employee|contractor)\b",
        ]

        self.entity_patterns = {
            "RISK": self.risk_patterns,
            "CONTROL": self.control_patterns,
            "ASSET": self.asset_patterns,
            "IMPACT": self.impact_patterns,
            "STAKEHOLDER": self.stakeholder_patterns,
            "COMPLIANCE": [
                r"\b(?:regulation|standard|policy|requirement|law|framework|mandate)\b",
                r"\b(?:compliance|regulatory|legal|statutory)\s+(?:requirement|obligation)\b",
                r"\b(?:GDPR|HIPAA|SOX|PCI DSS|ISO|NIST|Basel III)\b",
            ],
            "PROCESS": [
                r"\b(?:process|procedure|workflow|activity|operation|practice|methodology)\b",
                r"\b(?:risk assessment|risk management|incident response|business continuity)\b",
                r"\b(?:change management|access management|vulnerability management)\b",
            ],
            "THREAT": [
                r"\b(?:threat actor|attacker|adversary|malicious insider|hacker|cybercriminal)\b",
                r"\b(?:malware|ransomware|phishing|social engineering|DDoS|APT)\b",
            ],
            "VULNERABILITY": [
                r"\b(?:vulnerability|weakness|deficiency|gap|flaw|exposure|loophole)\b",
                r"\b(?:unpatched|outdated|misconfigured|unsecured|weak)\s+(?:system|software|control)\b",
            ],
        }

    def extract_entities(
            self, text: str, chunks: List[str]) -> List[Dict[str, Any]]:
        entities = []
        entity_map = defaultdict(
            lambda: {
                "count": 0,
                "contexts": [],
                "type": None,
                "pattern_matches": defaultdict(int),
                "spacy_labels": defaultdict(int),
                "positions": [],
            }
        )

        chunk_overlap = int(settings.CHUNK_OVERLAP)

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue

            chunk_to_process = chunk[: self.nlp.max_length]
            doc = self.nlp(chunk_to_process)

            for ent in doc.ents:
                entity_type = self._classify_entity_type(ent.text, ent.label_)
                if entity_type:
                    key = self._normalize_entity_text(ent.text)
                    entity_map[key]["count"] += 1
                    entity_map[key]["contexts"].append(
                        {
                            "text": chunk[:300],
                            "chunk_idx": chunk_idx,
                            "start": ent.start_char,
                            "end": ent.end_char,
                        }
                    )
                    entity_map[key]["type"] = entity_type
                    entity_map[key]["original_text"] = ent.text
                    entity_map[key]["spacy_labels"][ent.label_] += 1

            pattern_entities = self._extract_pattern_entities(chunk, chunk_idx)
            for entity in pattern_entities:
                key = self._normalize_entity_text(entity["text"])
                entity_map[key]["count"] += 1
                entity_map[key]["contexts"].append(
                    {
                        "text": entity["context"],
                        "chunk_idx": chunk_idx,
                        "pattern": entity["pattern"],
                    }
                )
                entity_map[key]["type"] = entity["type"]
                entity_map[key]["original_text"] = entity["text"]
                entity_map[key]["pattern_matches"][entity["pattern"]] += 1

        for key, data in entity_map.items():
            if data["count"] > 0 and data["type"]:
                confidence = self._calculate_confidence(data)

                entities.append(
                    {
                        "id": key.replace(" ", "_"),
                        "label": data["original_text"],
                        "type": data["type"],
                        "confidence": confidence,
                        "count": data["count"],
                        "contexts": [ctx["text"] for ctx in data["contexts"][:3]],
                        "metadata": {
                            "pattern_matches": dict(data["pattern_matches"]),
                            "spacy_labels": dict(data["spacy_labels"]),
                            "chunk_coverage": len(
                                set(ctx["chunk_idx"] for ctx in data["contexts"])
                            )
                            / len(chunks),
                        },
                    }
                )

        entities.sort(key=lambda x: x["confidence"], reverse=True)
        return entities

    def _normalize_entity_text(self, text: str) -> str:
        text = " ".join(text.split())
        return text.lower().strip()

    def _calculate_confidence(self, entity_data: Dict) -> float:
        count_score = min(
            math.log(
                entity_data["count"] +
                1) /
            math.log(10),
            1.0)

        pattern_diversity = len(entity_data["pattern_matches"]) / 5.0
        pattern_score = min(pattern_diversity, 1.0)

        spacy_score = 0.8 if entity_data["spacy_labels"] else 0.0

        unique_chunks = len(
            set(ctx.get("chunk_idx", 0) for ctx in entity_data["contexts"])
        )
        context_score = min(unique_chunks / 5.0, 1.0)

        confidence = (
            0.3 * count_score
            + 0.3 * pattern_score
            + 0.2 * spacy_score
            + 0.2 * context_score
        )

        if any(
            pattern in str(entity_data["pattern_matches"])
            for pattern in ["ISO", "NIST", "GDPR", "critical", "high risk"]
        ):
            confidence = min(confidence * 1.2, 1.0)

        return round(confidence, 3)

    def _classify_entity_type(self, text: str, spacy_label: str) -> str:
        text_lower = text.lower()

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return entity_type

        label_mapping = {
            "ORG": "STAKEHOLDER",
            "PERSON": "STAKEHOLDER",
            "LAW": "COMPLIANCE",
            "MONEY": "IMPACT",
            "PERCENT": "IMPACT",
            "PRODUCT": "ASSET",
            "FAC": "ASSET",
            "LOC": "ASSET",
            "DATE": None,
            "TIME": None,
            "QUANTITY": "IMPACT",
            "CARDINAL": "IMPACT",
        }

        return label_mapping.get(spacy_label, None)

    def _extract_pattern_entities(
        self, text: str, chunk_idx: int
    ) -> List[Dict[str, str]]:
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                enhanced_pattern = r"\b" + pattern.strip(r"\b") + r"\b"
                matches = re.finditer(enhanced_pattern, text, re.IGNORECASE)

                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]

                    entities.append(
                        {
                            "text": match.group().strip(),
                            "type": entity_type,
                            "pattern": pattern,
                            "context": context,
                            "chunk_idx": chunk_idx,
                        }
                    )

        return entities

    def extract_relationships(
        self, text: str, entities: List[Dict]
    ) -> List[Dict[str, Any]]:
        relationships = []
        entity_dict = {e["id"]: e for e in entities}

        text_to_process = text[: self.nlp.max_length]
        doc = self.nlp(text_to_process)

        for sent in doc.sents:
            if len(sent.text.split()) < 5:
                continue

            sent_entities = []
            sent_text_lower = sent.text.lower()

            for entity in entities:
                if (
                    entity["label"].lower() in sent_text_lower
                    or entity["id"].replace("_", " ") in sent_text_lower
                ):
                    sent_entities.append(entity)

            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i + 1:]:
                    rel = self._extract_relationship_type(sent.text, e1, e2)
                    if rel:
                        relationships.append(
                            {
                                "source": e1["id"],
                                "target": e2["id"],
                                "type": rel["type"],
                                "confidence": rel["confidence"],
                                "evidence": rel.get("evidence", sent.text[:200]),
                            }
                        )

        inferred_rels = self._infer_risk_relationships(entities)
        relationships.extend(inferred_rels)

        relationships = self._deduplicate_relationships(relationships)

        return relationships

    def _extract_relationship_type(
        self, sentence: str, entity1: Dict, entity2: Dict
    ) -> Dict[str, Any]:
        sentence_lower = sentence.lower()
        e1_text = entity1["label"].lower()
        e2_text = entity2["label"].lower()

        patterns = [
            (
                "MITIGATES",
                [
                    "mitigate",
                    "reduce",
                    "control",
                    "manage",
                    "minimize",
                    "address",
                    "treat",
                ],
                0.9,
            ),
            (
                "IMPLEMENTS",
                [
                    "implement",
                    "execute",
                    "deploy",
                    "establish",
                    "apply",
                    "enforce",
                    "use",
                ],
                0.85,
            ),
            (
                "CAUSES",
                [
                    "cause",
                    "lead to",
                    "result in",
                    "trigger",
                    "create",
                    "generate",
                    "produce",
                ],
                0.9,
            ),
            (
                "AFFECTS",
                ["affect", "impact", "influence",
                    "compromise", "threaten", "endanger"],
                0.85,
            ),
            (
                "EXPLOITS",
                ["exploit", "leverage", "utilize", "take advantage", "abuse"],
                0.9,
            ),
            (
                "OWNS",
                [
                    "own",
                    "responsible for",
                    "manage",
                    "oversee",
                    "accountable",
                    "administer",
                ],
                0.85,
            ),
            (
                "MONITORS",
                [
                    "monitor",
                    "track",
                    "observe",
                    "measure",
                    "assess",
                    "watch",
                    "supervise",
                ],
                0.8,
            ),
            (
                "REQUIRES",
                [
                    "require",
                    "need",
                    "depend",
                    "necessitate",
                    "must have",
                    "prerequisite",
                ],
                0.85,
            ),
            (
                "SUPPORTS",
                ["support", "enable", "facilitate", "assist", "help", "aid"],
                0.8,
            ),
            (
                "COMPLIES_WITH",
                ["comply with", "adhere to", "follow",
                    "meet", "satisfy", "conform"],
                0.9,
            ),
            (
                "VIOLATES",
                ["violate", "breach", "break", "non-compliant", "fail to meet"],
                0.9,
            ),
        ]

        for rel_type, keywords, base_confidence in patterns:
            for keyword in keywords:
                if keyword in sentence_lower:
                    if e1_text in sentence_lower and e2_text in sentence_lower:
                        e1_pos = sentence_lower.find(e1_text)
                        e2_pos = sentence_lower.find(e2_text)
                        keyword_pos = sentence_lower.find(keyword)

                        distance_factor = self._calculate_proximity_score(
                            e1_pos, e2_pos, keyword_pos, len(sentence)
                        )
                        confidence = base_confidence * distance_factor

                        if self._should_swap_direction(
                            rel_type, entity1["type"], entity2["type"]
                        ):
                            return {
                                "type": rel_type,
                                "confidence": confidence,
                                "evidence": sentence[:200],
                            }

                        return {
                            "type": rel_type,
                            "confidence": confidence,
                            "evidence": sentence[:200],
                        }

        return None

    def _calculate_proximity_score(
        self, pos1: int, pos2: int, keyword_pos: int, text_len: int
    ) -> float:
        max_distance = min(100, text_len // 2)

        entity_distance = abs(pos2 - pos1)

        keyword_distance = min(abs(keyword_pos - pos1),
                               abs(keyword_pos - pos2))

        entity_score = max(0, 1 - (entity_distance / max_distance))
        keyword_score = max(0, 1 - (keyword_distance / max_distance))

        return (entity_score + keyword_score) / 2

    def _should_swap_direction(
            self,
            rel_type: str,
            type1: str,
            type2: str) -> bool:
        expected_directions = {
            "MITIGATES": ("CONTROL", "RISK"),
            "AFFECTS": ("RISK", "ASSET"),
            "OWNS": ("STAKEHOLDER", "ASSET"),
            "IMPLEMENTS": ("STAKEHOLDER", "CONTROL"),
            "REQUIRES": ("CONTROL", "ASSET"),
            "EXPLOITS": ("THREAT", "VULNERABILITY"),
            "CAUSES": ("THREAT", "RISK"),
        }

        if rel_type in expected_directions:
            expected_source, expected_target = expected_directions[rel_type]
            if type1 == expected_target and type2 == expected_source:
                return True

        return False

    def _infer_risk_relationships(self, entities: List[Dict]) -> List[Dict]:
        relationships = []

        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity["type"]].append(entity)

        for risk in entities_by_type.get("RISK", []):
            for control in entities_by_type.get("CONTROL", []):
                if self._entities_potentially_related(risk, control):
                    relationships.append(
                        {
                            "source": control["id"],
                            "target": risk["id"],
                            "type": "MITIGATES",
                            "confidence": 0.7,
                            "evidence": "Inferred from entity types and context",
                        })

        for threat in entities_by_type.get("THREAT", []):
            for vuln in entities_by_type.get("VULNERABILITY", []):
                if self._entities_potentially_related(threat, vuln):
                    relationships.append(
                        {
                            "source": threat["id"],
                            "target": vuln["id"],
                            "type": "EXPLOITS",
                            "confidence": 0.75,
                            "evidence": "Inferred from threat-vulnerability relationship",
                        })

        for vuln in entities_by_type.get("VULNERABILITY", []):
            for risk in entities_by_type.get("RISK", []):
                if self._entities_potentially_related(vuln, risk):
                    relationships.append(
                        {
                            "source": vuln["id"],
                            "target": risk["id"],
                            "type": "CAUSES",
                            "confidence": 0.7,
                            "evidence": "Inferred from vulnerability-risk relationship",
                        })

        return relationships

    def _entities_potentially_related(
            self, entity1: Dict, entity2: Dict) -> bool:
        contexts1 = set(" ".join(entity1.get("contexts", [])).lower().split())
        contexts2 = set(" ".join(entity2.get("contexts", [])).lower().split())

        intersection = len(contexts1 & contexts2)
        union = len(contexts1 | contexts2)

        if union == 0:
            return False

        similarity = intersection / union

        if "metadata" in entity1 and "metadata" in entity2:
            pass

        return similarity > 0.1

    def _deduplicate_relationships(
            self, relationships: List[Dict]) -> List[Dict]:
        unique_rels = {}

        for rel in relationships:
            key = (rel["source"], rel["target"], rel["type"])

            if key in unique_rels:
                existing = unique_rels[key]
                if rel["confidence"] > existing["confidence"]:
                    unique_rels[key] = rel
                elif "evidence" in rel and rel["evidence"] not in existing.get(
                    "evidence", ""
                ):
                    existing["evidence"] += " | " + rel["evidence"]
            else:
                unique_rels[key] = rel

        return list(unique_rels.values())
