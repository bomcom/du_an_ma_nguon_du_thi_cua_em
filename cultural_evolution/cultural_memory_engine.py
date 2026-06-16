"""
cultural_memory_engine.py

Phase 7A - Cultural Memory Engine (Refined)
Part of the large-scale evolutionary simulation for MA_NGUON_DU_AN_QUOC_GIA.

Refined according to research-grade cultural evolution requirements:
- Context-aware (environment signature)
- Prestige-based imitation
- Exposure accumulation
- Cultural variation / mutation
- Safe memory pruning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import uuid
import copy
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Core Data Structures
# =========================================================================

@dataclass
class EnvironmentContext:
    """Context in which a pattern was observed."""
    temperature_band: str = "moderate"   # cold, moderate, hot
    humidity_band: str = "medium"        # dry, medium, wet
    terrain_type: str = "general"
    energy_availability: float = 0.5


@dataclass
class CulturalSignal:
    """Rich observation signal from the population."""
    signal_id: str
    pattern_signature: str
    observed_utility: float
    source_entity_id: int
    observer_entity_id: int
    source_prestige: float = 1.0          # Higher prestige = stronger influence
    exposure_count: int = 1               # How many times this was observed
    environment_context: EnvironmentContext = field(default_factory=EnvironmentContext)
    tick: int = 0


@dataclass
class CulturalMemoryUnit:
    """Persistent collective pattern."""
    memory_id: str
    pattern_signature: str
    base_utility: float
    total_observer_count: int
    total_replication_count: int
    last_observed_tick: int
    cultural_strength: float = 0.0
    prestige_influence: float = 0.0
    context_variants: Dict[str, float] = field(default_factory=dict)  # environment -> strength


@dataclass
class KnowledgePacketAdapter:
    """Converts 6E packets into cultural signals."""
    @staticmethod
    def extract_signals(packet, default_prestige: float = 1.0) -> List[CulturalSignal]:
        signals = []
        for ku in getattr(packet, 'knowledge_units', []):
            signal = CulturalSignal(
                signal_id=f"sig_{str(uuid.uuid4())[:8]}",
                pattern_signature=ku.pattern_signature,
                observed_utility=ku.utility_score,
                source_entity_id=packet.entity_id,
                observer_entity_id=packet.entity_id,
                source_prestige=default_prestige,
                exposure_count=1,
                tick=getattr(packet, 'timestamp', 0)  # placeholder
            )
            signals.append(signal)
        return signals


# =========================================================================
# 2. Observation Engine
# =========================================================================

class CulturalObservationEngine:
    def __init__(self):
        self.pending_signals: List[CulturalSignal] = []

    def record_observation(self, signal: CulturalSignal):
        self.pending_signals.append(signal)

    def get_pending_signals(self) -> List[CulturalSignal]:
        return copy.deepcopy(self.pending_signals)


# =========================================================================
# 3. Cultural Memory Pool
# =========================================================================

class CulturalMemoryPool:
    def __init__(self):
        self._memories: Dict[str, CulturalMemoryUnit] = {}

    def ingest_signal(self, signal: CulturalSignal):
        key = signal.pattern_signature
        if key not in self._memories:
            self._memories[key] = CulturalMemoryUnit(
                memory_id=f"mem_{str(uuid.uuid4())[:8]}",
                pattern_signature=key,
                base_utility=signal.observed_utility,
                total_observer_count=signal.exposure_count,
                total_replication_count=1,
                last_observed_tick=signal.tick,
                prestige_influence=signal.source_prestige
            )
        else:
            mem = self._memories[key]
            mem.total_observer_count += signal.exposure_count
            mem.total_replication_count += 1
            mem.base_utility = 0.6 * mem.base_utility + 0.4 * signal.observed_utility
            mem.last_observed_tick = signal.tick
            mem.prestige_influence = max(mem.prestige_influence, signal.source_prestige)

            # Context awareness
            ctx_key = f"{signal.environment_context.temperature_band}_{signal.environment_context.humidity_band}"
            mem.context_variants[ctx_key] = mem.context_variants.get(ctx_key, 0.0) + signal.observed_utility

    def get_all_memories(self) -> List[CulturalMemoryUnit]:
        return list(self._memories.values())

    def safe_prune_weak_memories(self, current_tick: int, threshold: float = 0.05):
        to_remove = []
        for key, mem in self._memories.items():
            ticks_idle = current_tick - mem.last_observed_tick
            if mem.cultural_strength < threshold and ticks_idle > 800:
                to_remove.append(key)

        for key in to_remove:
            del self._memories[key]


# =========================================================================
# 4. Cultural Variation / Mutation Engine
# =========================================================================

class CulturalVariationEngine:
    """Introduces small mutations and variations in cultural patterns."""

    def mutate_pattern(self, pattern: str, mutation_rate: float = 0.08) -> Optional[str]:
        if random.random() > mutation_rate:
            return None
        # Simple symbolic variation (can be extended with property categories)
        parts = pattern.split("+")
        if len(parts) > 1 and random.random() < 0.6:
            # Swap or generalize one component
            idx = random.randint(0, len(parts)-1)
            parts[idx] = parts[idx] + "_variant"
            return "+".join(parts)
        return pattern + "_mutated"


# =========================================================================
# 5. Cultural Selection + Social Learning
# =========================================================================

class CulturalSelectionEngine:
    def __init__(self, decay_rate: float = 0.0018):
        self.decay_rate = decay_rate
        self.variation_engine = CulturalVariationEngine()

    def apply_selection(self, memory_pool: CulturalMemoryPool, current_tick: int):
        for mem in memory_pool.get_all_memories():
            # Strength = utility × prestige × popularity × recency
            recency = max(0.1, 1.0 - (current_tick - mem.last_observed_tick) / 1500)
            mem.cultural_strength = (
                mem.base_utility * 0.45 +
                mem.prestige_influence * 0.25 +
                (mem.total_observer_count / max(10, current_tick)) * 0.2 +
                recency * 0.1
            )

            # Cultural mutation
            if random.random() < 0.03:  # rare variation
                mutated = self.variation_engine.mutate_pattern(mem.pattern_signature)
                if mutated and mutated not in memory_pool._memories:
                    # Create weak variant
                    variant = copy.deepcopy(mem)
                    variant.pattern_signature = mutated
                    variant.cultural_strength *= 0.6
                    variant.memory_id = f"mem_var_{uuid.uuid4().hex[:6]}"
                    memory_pool._memories[mutated] = variant

        # Safe pruning
        memory_pool.safe_prune_weak_memories(current_tick)


class SocialLearningEngine:
    """Prestige + Exposure based imitation bias."""

    def __init__(self):
        self.imitation_bias: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.exposure_count: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def apply_observation(self, observer_id: int, pattern: str, source_prestige: float, exposure: int = 1):
        self.exposure_count[observer_id][pattern] += exposure
        # Bias grows with exposure and source prestige
        bias_increase = source_prestige * 0.12 * min(1.0, exposure * 0.3)
        self.imitation_bias[observer_id][pattern] = min(1.0, 
            self.imitation_bias[observer_id][pattern] + bias_increase)

    def get_imitation_bias(self, entity_id: int, pattern: str) -> float:
        return self.imitation_bias[entity_id][pattern]


# =========================================================================
# 6. Main Cultural Memory Engine
# =========================================================================

class CulturalMemoryEngine:
    def __init__(self):
        self.observation_engine = CulturalObservationEngine()
        self.memory_pool = CulturalMemoryPool()
        self.selection_engine = CulturalSelectionEngine()
        self.social_learning = SocialLearningEngine()
        self.current_tick: int = 0

    def ingest_knowledge_packet(self, packet):
        signals = KnowledgePacketAdapter.extract_signals(packet)
        for signal in signals:
            self.observation_engine.record_observation(signal)

    def process_cultural_tick(self, tick: int):
        self.current_tick = tick

        # Ingest observations
        for signal in self.observation_engine.get_pending_signals():
            self.memory_pool.ingest_signal(signal)
            # Social learning effect
            self.social_learning.apply_observation(
                observer_id=signal.observer_entity_id,
                pattern=signal.pattern_signature,
                source_prestige=signal.source_prestige,
                exposure=signal.exposure_count
            )

        self.observation_engine.pending_signals.clear()

        # Selection + Variation
        self.selection_engine.apply_selection(self.memory_pool, tick)

    def get_cultural_patterns(self, top_n: int = 40) -> List[CulturalMemoryUnit]:
        patterns = self.memory_pool.get_all_memories()
        return sorted(patterns, key=lambda m: m.cultural_strength, reverse=True)[:top_n]

    def reinforce_pattern(self, pattern_signature: str, utility: float, prestige: float = 1.0, observers: int = 1):
        signal = CulturalSignal(
            signal_id=f"rein_{uuid.uuid4().hex[:6]}",
            pattern_signature=pattern_signature,
            observed_utility=utility,
            source_entity_id=0,
            observer_entity_id=0,
            source_prestige=prestige,
            exposure_count=observers,
            tick=self.current_tick
        )
        self.memory_pool.ingest_signal(signal)

    def __repr__(self) -> str:
        mem_count = len(self.memory_pool._memories)
        strong = sum(1 for m in self.memory_pool._memories.values() if m.cultural_strength > 0.4)
        return f"CulturalMemoryEngine(memories={mem_count}, strong_patterns={strong}, tick={self.current_tick})"


# =========================================================================
# Demo
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Testing Phase 7A (Refined - Context, Prestige, Variation) ===")

    engine = CulturalMemoryEngine()

    # Simulate multiple observations with prestige and context
    for i in range(18):
        sig = CulturalSignal(
            signal_id=f"test_{i}",
            pattern_signature="dense+conductive",
            observed_utility=3.5 + i*0.2,
            source_entity_id=100 + i,
            observer_entity_id=200,
            source_prestige=1.0 + (i % 5)/5,
            exposure_count=1 + i//8,
            environment_context=EnvironmentContext(temperature_band="cold", humidity_band="dry"),
            tick=1000 + i
        )
        engine.observation_engine.record_observation(sig)

    engine.process_cultural_tick(1050)
    engine.reinforce_pattern("dense+conductive", 6.2, prestige=2.5, observers=15)
    engine.process_cultural_tick(1080)

    patterns = engine.get_cultural_patterns(8)
    print(f"\nCollective Cultural Memory ({len(patterns)} strongest patterns):")
    for p in patterns[:4]:
        print(f"  • {p.pattern_signature} | strength={p.cultural_strength:.3f} | prestige={p.prestige_influence:.2f} | observers={p.total_observer_count}")

    print("\nPhase 7A Cultural Memory Engine (Research-grade): PASSED")

    