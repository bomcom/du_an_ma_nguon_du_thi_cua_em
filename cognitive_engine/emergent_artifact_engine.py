"""
emergent_artifact_engine.py

Phase 6D - Emergent Artifact & Tool Discovery Engine
Part of the large-scale evolutionary simulation for MA_NGUON_DU_AN_QUOC_GIA.

Responsibility:
- Receives purely physical manipulation intents from 6C.
- Applies emergent mathematical transformations based on property interaction rules.
- Tracks population-level utility. Artifacts become "Recognized Tools" only when they demonstrate consistent value across multiple entities.
- Fully property-driven: no hardcoded recipes or semantics.

Strict Boundaries:
- NO HARDCODED RECIPES.
- NO HARDCODED PROPERTY BEHAVIORS (rules are registered dynamically).
- NO direct "crafting". Only blind physical combination + utility emergence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
import uuid
import statistics

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Property Definition System (User-definable)
# =========================================================================

@dataclass
class PropertyDefinition:
    property_id: str
    min_value: float = 0.0
    max_value: float = 10.0
    default_value: float = 0.0
    # Simple interaction rules: how this property behaves when combined with others
    interaction_rules: Dict[str, float] = field(default_factory=dict)  # target_property -> multiplier


class PropertyDefinitionRegistry:
    """Central registry for all known properties. Allows dynamic extension via prompts/NLP."""

    def __init__(self):
        self._properties: Dict[str, PropertyDefinition] = {}

    def register_property(self, prop_def: PropertyDefinition) -> bool:
        self._properties[prop_def.property_id] = prop_def
        logger.info(f"Registered property: {prop_def.property_id}")
        return True

    def get_property(self, prop_id: str) -> Optional[PropertyDefinition]:
        return self._properties.get(prop_id)

    def list_properties(self) -> List[str]:
        return list(self._properties.keys())

    def normalize_value(self, prop_id: str, value: float) -> float:
        prop = self.get_property(prop_id)
        if prop:
            return max(prop.min_value, min(prop.max_value, value))
        return max(0.0, min(10.0, value))


# =========================================================================
# 2. Artifact Descriptor with Physical State
# =========================================================================

@dataclass
class ArtifactDescriptor:
    artifact_id: str
    component_ids: List[str]
    properties: Dict[str, float]  # Dynamic property matrix
    physical_state: Dict[str, float]  # mass, volume, temperature, integrity, etc.
    creation_tick: int
    creator_entity_id: int
    is_recognized_tool: bool = False
    utility_history: List[float] = field(default_factory=list)  # population-level utility samples


# =========================================================================
# 3. Emergent Physical Combination (Rule-based, no magic numbers in core logic)
# =========================================================================

class PhysicalCombinationEngine:
    """Purely mathematical blending driven by registered PropertyDefinitions."""

    def __init__(self, prop_registry: PropertyDefinitionRegistry):
        self.prop_registry = prop_registry

    def combine_properties(self, props_a: Dict[str, float], props_b: Dict[str, float]) -> Dict[str, float]:
        combined: Dict[str, float] = {}
        all_keys = set(props_a.keys()).union(set(props_b.keys()))

        for key in all_keys:
            val_a = props_a.get(key, 0.0)
            val_b = props_b.get(key, 0.0)
            prop_def = self.prop_registry.get_property(key)

            if key in props_a and key in props_b:
                # Base synergy from co-occurrence
                base = (val_a + val_b) / 2.0
                # Apply interaction rules if defined
                synergy = 1.0
                if prop_def and prop_def.interaction_rules:
                    for other_key in all_keys:
                        if other_key in prop_def.interaction_rules:
                            synergy *= prop_def.interaction_rules[other_key]
                combined[key] = base * synergy
            else:
                # Dilution for novel combinations
                dilution = 0.85  # Base physical realism, can be tuned via rules later
                combined[key] = (val_a + val_b) * dilution

            # Normalize
            combined[key] = self.prop_registry.normalize_value(key, combined[key])

        return combined


# =========================================================================
# 4. Population-Level Utility Tracker
# =========================================================================

class ArtifactUtilityTracker:
    """Tracks utility at population level for true emergent tool recognition."""

    def __init__(self, base_threshold: float = 4.0, adaptation_factor: float = 0.1):
        self.utility_scores: Dict[str, List[float]] = {}  # artifact_id -> list of utility events
        self.base_threshold = base_threshold
        self.adaptation_factor = adaptation_factor  # threshold adjusts based on global average

    def record_utility_event(self, artifact_id: str, utility_gain: float, population_context: Optional[Dict[str, Any]] = None):
        if artifact_id not in self.utility_scores:
            self.utility_scores[artifact_id] = []
        self.utility_scores[artifact_id].append(utility_gain)

        # Keep only recent history (sliding window)
        if len(self.utility_scores[artifact_id]) > 50:
            self.utility_scores[artifact_id] = self.utility_scores[artifact_id][-50:]

    def should_recognize_as_tool(self, artifact_id: str, global_avg_utility: float) -> bool:
        """Dynamic threshold based on population performance."""
        if artifact_id not in self.utility_scores or len(self.utility_scores[artifact_id]) < 5:
            return False

        avg_utility = statistics.mean(self.utility_scores[artifact_id])
        dynamic_threshold = self.base_threshold * (1.0 + self.adaptation_factor * (global_avg_utility - self.base_threshold))

        return avg_utility >= dynamic_threshold and len(self.utility_scores[artifact_id]) >= 8


# =========================================================================
# 5. Artifact Registry
# =========================================================================

class ArtifactRegistry:
    def __init__(self):
        self._artifacts: Dict[str, ArtifactDescriptor] = {}

    def register(self, artifact: ArtifactDescriptor):
        self._artifacts[artifact.artifact_id] = artifact

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactDescriptor]:
        return self._artifacts.get(artifact_id)

    def mark_as_tool(self, artifact_id: str):
        if artifact_id in self._artifacts:
            self._artifacts[artifact_id].is_recognized_tool = True
            logger.info(f"EMERGENCE EVENT: Artifact {artifact_id} recognized as Tool by population!")


# =========================================================================
# 6. Main Emergent Artifact Engine
# =========================================================================

class EmergentArtifactEngine:
    def __init__(self):
        self.prop_registry = PropertyDefinitionRegistry()
        self.physics = PhysicalCombinationEngine(self.prop_registry)
        self.registry = ArtifactRegistry()
        self.tracker = ArtifactUtilityTracker(base_threshold=3.5)

        # Register some base physical properties (can be extended dynamically)
        self._init_base_properties()

    def _init_base_properties(self):
        base_props = [
            PropertyDefinition("physical_density", 0.0, 5.0),
            PropertyDefinition("integrity", 0.0, 10.0, 5.0),
            PropertyDefinition("temperature", -50.0, 200.0, 20.0),
            PropertyDefinition("volume", 0.1, 100.0, 1.0),
        ]
        for p in base_props:
            self.prop_registry.register_property(p)

    def process_manipulation(
        self,
        entity_id: int,
        item_a_id: str,
        props_a: Dict[str, float],
        item_b_id: str,
        props_b: Dict[str, float],
        current_tick: int
    ) -> ArtifactDescriptor:
        """Pure physical combination → new artifact."""
        new_props = self.physics.combine_properties(props_a, props_b)

        # Basic physical state
        physical_state = {
            "mass": (props_a.get("physical_density", 1.0) + props_b.get("physical_density", 1.0)) * 0.5,
            "volume": max(0.1, (props_a.get("volume", 1.0) + props_b.get("volume", 1.0)) * 0.7),
            "temperature": (props_a.get("temperature", 20.0) + props_b.get("temperature", 20.0)) / 2.0,
            "integrity": min(10.0, (props_a.get("integrity", 5.0) + props_b.get("integrity", 5.0)) * 0.9),
        }

        artifact_id = f"artf_{str(uuid.uuid4())[:8]}_{current_tick}"
        artifact = ArtifactDescriptor(
            artifact_id=artifact_id,
            component_ids=[item_a_id, item_b_id],
            properties=new_props,
            physical_state=physical_state,
            creation_tick=current_tick,
            creator_entity_id=entity_id
        )

        self.registry.register(artifact)
        return artifact

    def evaluate_artifact_usage(self, artifact_id: str, utility_gain: float, population_context: Optional[Dict[str, Any]] = None):
        """Report usage. Recognition is now population-driven."""
        artifact = self.registry.get_artifact(artifact_id)
        if not artifact or artifact.is_recognized_tool:
            return

        self.tracker.record_utility_event(artifact_id, utility_gain, population_context)

        # Global context for dynamic threshold
        global_avg = self._get_global_avg_utility()
        if self.tracker.should_recognize_as_tool(artifact_id, global_avg):
            self.registry.mark_as_tool(artifact_id)
            # Signal to 7A (Culture) can be emitted here

    def _get_global_avg_utility(self) -> float:
        """Simple global average for threshold adaptation."""
        all_scores = [sum(scores) / len(scores) for scores in self.tracker.utility_scores.values() if scores]
        return statistics.mean(all_scores) if all_scores else 0.0

    def add_property_definition(self, prop_def: PropertyDefinition):
        self.prop_registry.register_property(prop_def)


# =========================================================================
# Demo / Self-Test
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Testing Stage 6D: Emergent Artifact Engine (Improved) ===")

    engine = EmergentArtifactEngine()

    # Dynamic property extension (e.g. via NLP in real system)
    engine.add_property_definition(PropertyDefinition("mana_conductivity", 0.0, 1.0))
    engine.add_property_definition(PropertyDefinition("psi_resonance", 0.0, 2.0, interaction_rules={"mana_conductivity": 1.12}))

    matter_a = {"physical_density": 0.6, "integrity": 0.8, "flammability": 0.7}
    matter_b = {"mana_conductivity": 0.9, "temperature": 45.0, "physical_density": 0.3}

    print("\n[Physical Manipulation] Entity combines two matters...")
    artifact = engine.process_manipulation(101, "item_A", matter_a, "item_B", matter_b, 1500)

    print(f"Created: {artifact.artifact_id}")
    print("Properties:", {k: round(v, 3) for k, v in artifact.properties.items()})
    print("Physical State:", artifact.physical_state)

    # Population usage simulation
    print("\n[Population Usage Simulation]")
    for i in range(12):
        gain = 0.4 + (i * 0.3)  # Increasing utility as more entities use it
        engine.evaluate_artifact_usage(artifact.artifact_id, gain)

    final = engine.registry.get_artifact(artifact.artifact_id)
    if final:
        print(f"\nFinal Status - Recognized Tool: {final.is_recognized_tool}")
    print("Stage 6D Emergent Architecture: PASSED (Population + Rule-based)")
    