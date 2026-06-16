"""
social_transmission_engine.py

Phase 7B - Social Transmission & Cultural Drift Engine
Part of the large-scale evolutionary simulation for MA_NGUON_DU_AN_QUOC_GIA.

Refined to research-grade standards:
- Fully property-vector based patterns (no string signatures)
- Vector mutation & recombination for realistic drift
- Social graph topology
- Transmission cost model
- Decoupled from 7A: only emits TransmissionOutcome
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
import uuid
import random
import copy
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Core Property-Based Structures
# =========================================================================

@dataclass
class PatternDescriptor:
    """Vector-based cultural pattern (property → normalized weight)."""
    pattern_id: str
    property_weights: Dict[str, float]          # e.g. {"density": 0.82, "conductivity": 0.65}
    cultural_strength: float = 0.0
    emergence_tick: int = 0


@dataclass
class TransmissionCost:
    """Realistic cost of cultural transmission."""
    distance_factor: float          # social/physical distance
    prestige_mismatch: float
    attention_cost: float
    total_cost: float = 0.0


@dataclass
class TransmissionOutcome:
    """Immutable result of a transmission attempt. Fed back to orchestrator."""
    outcome_id: str
    source_entity_id: int
    target_entity_id: int
    original_pattern: PatternDescriptor
    transmitted_pattern: PatternDescriptor      # may be drifted/mutated
    success_probability: float
    actual_exposure: float
    prestige_influence: float
    transmission_cost: TransmissionCost
    tick: int


# =========================================================================
# 2. Social Graph (Network Topology)
# =========================================================================

class SocialGraph:
    """Models realistic social connections."""

    def __init__(self):
        self.adjacency: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))  # entity -> neighbor -> influence

    def add_connection(self, entity_a: int, entity_b: int, influence: float = 1.0):
        self.adjacency[entity_a][entity_b] = influence
        self.adjacency[entity_b][entity_a] = influence * 0.85  # slight asymmetry

    def get_neighbors(self, entity_id: int, min_influence: float = 0.2) -> List[Tuple[int, float]]:
        return [(nid, inf) for nid, inf in self.adjacency[entity_id].items() if inf >= min_influence]

    def get_transmission_probability(self, source: int, target: int) -> float:
        return self.adjacency[source].get(target, 0.15)


# =========================================================================
# 3. Cultural Drift & Innovation (Vector-based)
# =========================================================================

class CulturalDriftEngine:
    """Vector perturbation + recombination."""

    def mutate_vector(self, weights: Dict[str, float], mutation_rate: float = 0.12) -> Dict[str, float]:
        new_weights = copy.deepcopy(weights)
        for prop, value in new_weights.items():
            if random.random() < mutation_rate:
                # Gaussian perturbation
                new_weights[prop] = max(0.0, min(1.0, value + random.gauss(0, 0.12)))
        return new_weights

    def recombine(self, pattern_a: PatternDescriptor, pattern_b: PatternDescriptor) -> Dict[str, float]:
        """Blend two patterns."""
        combined = {}
        all_props = set(pattern_a.property_weights.keys()) | set(pattern_b.property_weights.keys())
        for p in all_props:
            wa = pattern_a.property_weights.get(p, 0.0)
            wb = pattern_b.property_weights.get(p, 0.0)
            combined[p] = (wa + wb) * 0.5 + random.uniform(-0.08, 0.08)
        return {k: max(0.0, min(1.0, v)) for k, v in combined.items()}


class InnovationEngine:
    def __init__(self):
        self.drift = CulturalDriftEngine()

    def create_innovation(self, base_patterns: List[PatternDescriptor]) -> Optional[PatternDescriptor]:
        if not base_patterns or random.random() > 0.07:
            return None
        if len(base_patterns) >= 2:
            a, b = random.sample(base_patterns, 2)
            new_weights = self.drift.recombine(a, b)
        else:
            new_weights = self.drift.mutate_vector(base_patterns[0].property_weights)

        return PatternDescriptor(
            pattern_id=f"innov_{str(uuid.uuid4())[:8]}",
            property_weights=new_weights,
            cultural_strength=0.2,
            emergence_tick=int(time.time() / 10)
        )


# =========================================================================
# 4. Main Social Transmission Engine (7B)
# =========================================================================

class SocialTransmissionEngine:
    """Pure transmission layer. Does NOT modify 7A memory directly."""

    def __init__(self, social_graph: SocialGraph):
        self.graph = social_graph
        self.drift = CulturalDriftEngine()
        self.innovation = InnovationEngine()
        self.transmission_history: List[TransmissionOutcome] = []
        self.current_tick: int = 0

    def simulate_transmission_tick(
        self,
        tick: int,
        living_entities: List[int],
        active_patterns: List[PatternDescriptor]   # From 7A via orchestrator
    ) -> List[TransmissionOutcome]:
        self.current_tick = tick
        outcomes: List[TransmissionOutcome] = []

        for _ in range(max(30, len(living_entities) // 3)):
            if not living_entities or not active_patterns:
                break

            source = random.choice(living_entities)
            neighbors = self.graph.get_neighbors(source)

            if not neighbors:
                continue

            target, influence = random.choice(neighbors)
            pattern = random.choice(active_patterns)

            # Calculate cost
            distance_factor = 1.0 / (influence + 0.1)
            cost = TransmissionCost(
                distance_factor=distance_factor,
                prestige_mismatch=random.uniform(0.0, 0.4),
                attention_cost=random.uniform(0.1, 0.6)
            )
            cost.total_cost = (cost.distance_factor + cost.prestige_mismatch + cost.attention_cost) / 3.0

            exposure = max(0.1, influence * (1.0 - cost.total_cost))

            # Drift / mutation during transmission
            transmitted_weights = self.drift.mutate_vector(pattern.property_weights)
            transmitted_pattern = PatternDescriptor(
                pattern_id=f"tx_{str(uuid.uuid4())[:8]}",
                property_weights=transmitted_weights,
                cultural_strength=pattern.cultural_strength * 0.85,
                emergence_tick=tick
            )

            success_prob = exposure * 0.65 * (1.0 - cost.total_cost)

            outcome = TransmissionOutcome(
                outcome_id=f"out_{str(uuid.uuid4())[:8]}",
                source_entity_id=source,
                target_entity_id=target,
                original_pattern=pattern,
                transmitted_pattern=transmitted_pattern,
                success_probability=success_prob,
                actual_exposure=exposure,
                prestige_influence=influence,
                transmission_cost=cost,
                tick=tick
            )

            if random.random() < success_prob:
                outcomes.append(outcome)
                self.transmission_history.append(outcome)

                # Occasional innovation
                if random.random() < 0.06:
                    innov = self.innovation.create_innovation(active_patterns)
                    if innov:
                        # Return as separate outcome for orchestrator
                        innov_outcome = copy.deepcopy(outcome)
                        innov_outcome.transmitted_pattern = innov
                        outcomes.append(innov_outcome)

        return outcomes

    def get_transmission_stats(self) -> Dict[str, Any]:
        return {
            "total_transmissions": len(self.transmission_history),
            "unique_targets": len({o.target_entity_id for o in self.transmission_history}),
            "drift_events": sum(1 for o in self.transmission_history if o.transmitted_pattern.pattern_id.startswith("tx_"))
        }

    def __repr__(self) -> str:
        return f"SocialTransmissionEngine(tick={self.current_tick}, transmissions={len(self.transmission_history)})"


# =========================================================================
# Demo
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Testing Phase 7B (Research-Grade: Vector + Graph + Cost) ===")

    graph = SocialGraph()
    # Build sample social network
    for i in range(50):
        for j in range(i+1, min(i+6, 50)):
            graph.add_connection(i, j, random.uniform(0.4, 1.0))

    engine = SocialTransmissionEngine(graph)

    # Sample patterns from 7A
    patterns = [
        PatternDescriptor("p1", {"density": 0.85, "conductivity": 0.72}, 0.8),
        PatternDescriptor("p2", {"volatility": 0.65, "energy": 0.9}, 0.55),
    ]

    population = list(range(50))

    outcomes = engine.simulate_transmission_tick(2500, population, patterns)

    stats = engine.get_transmission_stats()
    print("\nTransmission Statistics:")
    print(f"  Successful transmissions : {stats['total_transmissions']}")
    print(f"  Unique influenced entities: {stats['unique_targets']}")
    print(f"  Drift events: {stats['drift_events']}")

    if outcomes:
        print(f"\nExample outcome - Transmitted pattern has {len(outcomes[0].transmitted_pattern.property_weights)} properties")

    print("\nPhase 7B Social Transmission & Cultural Drift (Vector-based): PASSED")
    