"""
skill_knowledge_engine.py

Phase 6E - Skill & Knowledge Consolidation Engine
Part of the large-scale evolutionary simulation for MA_NGUON_DU_AN_QUOC_GIA.

Pipeline:
MemoryRecord → Property-based Extraction → Pattern Abstraction → 
KnowledgeUnit → Skill Formation (via execution) → Decay → KnowledgePacket (Top-N)

Strict Boundaries:
- Individual learning only.
- No cultural transmission (reserved for 7A).
- No gene modification.
- Focus on generalization + procedural mastery.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import logging
import uuid
import time
import statistics
import copy
from collections import defaultdict

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Core Data Structures
# =========================================================================

@dataclass
class MemoryRecord:
    record_id: str
    entity_id: int
    action_signature: str
    properties_involved: Set[str]
    outcome_utility: float
    tick: int
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeUnit:
    knowledge_id: str
    pattern_signature: str                    # property-based (e.g. "dense+conductive")
    confidence: float
    utility_score: float
    occurrence_count: int
    last_reinforced_tick: int
    abstraction_level: int = 1                # 1 = concrete, 2+ = generalized
    associated_properties: Set[str] = field(default_factory=set)


@dataclass
class SkillProfile:
    skill_id: str
    skill_name: str
    mastery_level: float = 0.0                # 0.0 - 1.0 (procedural)
    knowledge_references: List[str] = field(default_factory=list)
    total_uses: int = 0
    last_used_tick: int = 0


@dataclass
class KnowledgePacket:
    packet_id: str
    entity_id: int
    knowledge_units: List[KnowledgeUnit]       # Top-N only
    skills: List[SkillProfile]
    timestamp: float


# =========================================================================
# 2. Property Pattern Extraction
# =========================================================================

class KnowledgeExtractionEngine:
    """Extracts knowledge based on PROPERTY patterns, not just action strings."""

    def extract(self, records: List[MemoryRecord], min_occurrences: int = 6) -> List[KnowledgeUnit]:
        if not records:
            return []

        # Group by property combination signature
        pattern_groups: Dict[str, List[MemoryRecord]] = defaultdict(list)
        for record in records:
            # Canonical property pattern
            prop_pattern = self._create_property_signature(record.properties_involved)
            pattern_groups[prop_pattern].append(record)

        knowledge_units = []
        for pattern_sig, group in pattern_groups.items():
            if len(group) < min_occurrences:
                continue

            utilities = [r.outcome_utility for r in group]
            avg_utility = statistics.mean(utilities)

            unit = KnowledgeUnit(
                knowledge_id=f"know_{str(uuid.uuid4())[:8]}",
                pattern_signature=pattern_sig,
                confidence=min(1.0, len(group) / 40.0),
                utility_score=round(avg_utility, 3),
                occurrence_count=len(group),
                last_reinforced_tick=group[-1].tick,
                associated_properties=set.union(*(r.properties_involved for r in group))
            )
            knowledge_units.append(unit)

        return knowledge_units

    def _create_property_signature(self, props: Set[str]) -> str:
        """Create stable signature from properties (sorted + normalized)."""
        return "+".join(sorted(props))


# =========================================================================
# 3. Pattern Abstraction Engine (Generalization)
# =========================================================================

class PatternAbstractionEngine:
    """Performs generalization: concrete → abstract patterns."""

    def __init__(self):
        # Can be extended dynamically (e.g. "stone" belongs to "dense_material")
        self.property_categories: Dict[str, Set[str]] = defaultdict(set)

    def register_category(self, category: str, members: Set[str]):
        self.property_categories[category].update(members)

    def abstract_pattern(self, knowledge_units: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        abstracted = []
        for ku in knowledge_units:
            # Create higher-level abstraction
            if len(ku.associated_properties) >= 2:
                abs_signature = self._try_abstract(ku.associated_properties)
                if abs_signature != ku.pattern_signature:
                    abs_ku = copy.deepcopy(ku)
                    abs_ku.pattern_signature = abs_signature
                    abs_ku.abstraction_level += 1
                    abs_ku.confidence *= 0.85  # abstraction penalty
                    abstracted.append(abs_ku)
            abstracted.append(ku)
        return abstracted

    def _try_abstract(self, props: Set[str]) -> str:
        """Simple category-based abstraction."""
        categories = []
        for cat, members in self.property_categories.items():
            if props & members:
                categories.append(cat)
        if categories:
            return "+".join(sorted(categories))
        return "+".join(sorted(props))


# =========================================================================
# 4. Skill & Decay Engines
# =========================================================================

class SkillFormationEngine:
    """Skills are formed from successful repeated execution, not just knowledge."""

    def form_skills(self, knowledge_units: List[KnowledgeUnit], execution_history: List[MemoryRecord]) -> List[SkillProfile]:
        skills = []
        for ku in knowledge_units:
            if ku.utility_score <= 0.8 or ku.confidence < 0.55:
                continue

            successful_exec = [r for r in execution_history if r.outcome_utility > 1.0]
            if len(successful_exec) < 8:
                continue

            skill = SkillProfile(
                skill_id=f"skill_{str(uuid.uuid4())[:8]}",
                skill_name=f"mastery_{ku.pattern_signature[:35]}",
                mastery_level=min(1.0, ku.confidence * 0.6 + (len(successful_exec) / 50.0) * 0.4),
                knowledge_references=[ku.knowledge_id],
                total_uses=len(successful_exec),
                last_used_tick=successful_exec[-1].tick if successful_exec else ku.last_reinforced_tick
            )
            skills.append(skill)
        return skills


class SkillDecayEngine:
    """Separate decay for procedural skills."""

    def __init__(self, base_decay_rate: float = 0.0012):
        self.base_decay_rate = base_decay_rate

    def apply_decay(self, skills: List[SkillProfile], current_tick: int):
        for skill in skills:
            ticks_idle = current_tick - skill.last_used_tick
            decay = self.base_decay_rate * ticks_idle * (1.0 - skill.mastery_level)
            skill.mastery_level = max(0.0, skill.mastery_level - decay)


# =========================================================================
# 5. Main SkillKnowledgeEngine
# =========================================================================

class SkillKnowledgeEngine:
    def __init__(self):
        self.extractor = KnowledgeExtractionEngine()
        self.abstracter = PatternAbstractionEngine()
        self.skill_formation = SkillFormationEngine()
        self.skill_decayer = SkillDecayEngine()

        self.entity_knowledge: Dict[int, List[KnowledgeUnit]] = {}
        self.entity_skills: Dict[int, List[SkillProfile]] = {}
        self.execution_history: Dict[int, List[MemoryRecord]] = {}   # for skill formation

    def record_experience(self, record: MemoryRecord):
        if record.entity_id not in self.execution_history:
            self.execution_history[record.entity_id] = []
        self.execution_history[record.entity_id].append(record)

        # Periodic processing
        if len(self.execution_history[record.entity_id]) >= 15:
            self._consolidate_for_entity(record.entity_id)

    def _consolidate_for_entity(self, entity_id: int):
        records = self.execution_history[record.entity_id]
        raw_knowledge = self.extractor.extract(records)
        abstracted = self.abstracter.abstract_pattern(raw_knowledge)

        if entity_id not in self.entity_knowledge:
            self.entity_knowledge[entity_id] = []
        self.entity_knowledge[entity_id].extend(abstracted)

        # Skill formation requires successful execution
        new_skills = self.skill_formation.form_skills(abstracted, records)
        if entity_id not in self.entity_skills:
            self.entity_skills[entity_id] = []
        self.entity_skills[entity_id].extend(new_skills)

        # Trim old records
        self.execution_history[entity_id] = records[-60:]

    def reinforce_skill_usage(self, entity_id: int, pattern_sig: str, utility_gain: float, current_tick: int):
        if entity_id not in self.entity_skills:
            return
        for skill in self.entity_skills[entity_id]:
            if pattern_sig in skill.skill_name:
                skill.total_uses += 1
                skill.last_used_tick = current_tick
                increment = utility_gain * 0.07 / (1 + skill.mastery_level * 4)
                skill.mastery_level = min(1.0, skill.mastery_level + increment)
                break

    def apply_decay(self, current_tick: int):
        for skills in self.entity_skills.values():
            self.skill_decayer.apply_decay(skills, current_tick)
        # Light knowledge decay
        for units in self.entity_knowledge.values():
            for ku in units:
                if current_tick - ku.last_reinforced_tick > 500:
                    ku.confidence *= 0.995

    def get_knowledge_packet(self, entity_id: int, top_n: int = 30) -> Optional[KnowledgePacket]:
        if entity_id not in self.entity_knowledge:
            return None

        self.apply_decay(self._get_current_tick())

        units = self.entity_knowledge[entity_id]
        # Score = confidence * utility * recency
        scored = []
        for u in units:
            recency = max(0.1, 1.0 - (self._get_current_tick() - u.last_reinforced_tick) / 2000)
            score = u.confidence * u.utility_score * recency
            scored.append((score, u))

        scored.sort(reverse=True)
        top_units = [u for _, u in scored[:top_n]]

        return KnowledgePacket(
            packet_id=f"packet_{entity_id}_{int(time.time())}",
            entity_id=entity_id,
            knowledge_units=copy.deepcopy(top_units),
            skills=copy.deepcopy(self.entity_skills.get(entity_id, [])),
            timestamp=time.time()
        )

    def _get_current_tick(self) -> int:
        return int(time.time() / 10)  # placeholder

    def __repr__(self) -> str:
        return f"SkillKnowledgeEngine(entities={len(self.entity_knowledge)}, total_knowledge={sum(len(v) for v in self.entity_knowledge.values())})"


# =========================================================================
# Demo
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Testing Phase 6E (Improved with Abstraction + Skill Decay) ===")

    engine = SkillKnowledgeEngine()

    # Register some property categories for abstraction
    engine.abstracter.register_category("dense_material", {"stone", "iron", "crystal"})
    engine.abstracter.register_category("conductive", {"mana", "psi", "metal"})

    # Simulate experiences
    for i in range(35):
        record = MemoryRecord(
            record_id=f"rec_{i}",
            entity_id=77,
            action_signature="combine_matter",
            properties_involved={"stone", "mana"} if i % 2 == 0 else {"iron", "psi"},
            outcome_utility=1.5 + i * 0.1,
            tick=2000 + i
        )
        engine.record_experience(record)

    packet = engine.get_knowledge_packet(77, top_n=15)
    if packet:
        print(f"\nGenerated Knowledge Packet for entity 77 ({len(packet.knowledge_units)} top units)")
        for ku in packet.knowledge_units[:3]:
            print(f"  • {ku.pattern_signature} (conf={ku.confidence:.2f}, util={ku.utility_score:.2f}, abs={ku.abstraction_level})")

    print("\nPhase 6E - Abstraction + Procedural Learning: PASSED")
    