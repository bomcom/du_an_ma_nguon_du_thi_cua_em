"""
ecosystem_orchestrator.py

Phase 5C - Ecosystem Orchestrator
Pure coordinator. No data storage. No invented APIs.
"""

from dataclasses import dataclass
from typing import Dict, Any
import logging

from .ecological_niche_manager import EcologicalNicheManager
from .environment_pressure_engine import EnvironmentPressureEngine

logger = logging.getLogger(__name__)


@dataclass
class SimulationSnapshot:
    tick: int
    species_count: int
    population_count: int
    active_pressures: int
    avg_adaptation_score: float = 0.0


class EcosystemOrchestrator:
    def __init__(
        self,
        niche_manager: EcologicalNicheManager,
        pressure_engine: EnvironmentPressureEngine,
        selection_engine=None,   # 4A
        genome_engine=None,      # 4B
        lineage_tracker=None,    # 4C
        species_manager=None     # 4D
    ):
        self.niche_manager = niche_manager
        self.pressure_engine = pressure_engine
        self.selection_engine = selection_engine
        self.genome_engine = genome_engine
        self.lineage_tracker = lineage_tracker
        self.species_manager = species_manager
        self.current_tick: int = 0

    def evolution_tick(self, environment_state: Dict[str, Any]) -> SimulationSnapshot:
        self.current_tick += 1

        species_ids = self.species_manager.list_species() if self.species_manager else set()

        # 5B: Generate pressures
        new_pressures = self.pressure_engine.generate_pressures(
            species_ids=species_ids,
            niche_manager=self.niche_manager,
            environment_state=environment_state
        )

        # 4A: Selection reacts to pressures (nếu có)
        if self.selection_engine and hasattr(self.selection_engine, 'evaluate_pressures'):
            self.selection_engine.evaluate_pressures(new_pressures)

        # 4B + 4A: Reproduction (placeholder)
        if self.selection_engine and self.genome_engine and hasattr(self.selection_engine, 'reproduce_generation'):
            offspring_queue = self.selection_engine.reproduce_generation()
            if self.genome_engine and hasattr(self.genome_engine, 'spawn_offspring_generation'):
                self.genome_engine.spawn_offspring_generation(offspring_queue)

        # 4C + 4D: Update tracking
        if self.lineage_tracker and hasattr(self.lineage_tracker, 'update'):
            self.lineage_tracker.update()
        if self.species_manager and hasattr(self.species_manager, 'update'):
            self.species_manager.update()

        # Snapshot
        avg_adaptation = self._safe_avg_adaptation(species_ids, environment_state)

        snapshot = SimulationSnapshot(
            tick=self.current_tick,
            species_count=len(species_ids),
            population_count=0,  # Will be wired to ECS later
            active_pressures=len(self.pressure_engine.get_active_pressures()),
            avg_adaptation_score=avg_adaptation
        )

        return snapshot

    def _safe_avg_adaptation(self, species_ids, environment_state) -> float:
        if not species_ids:
            return 0.0
        scores = [self.niche_manager.calculate_adaptation_score(sid, environment_state) for sid in species_ids]
        return sum(scores) / len(scores)

    def __repr__(self) -> str:
        return f"EcosystemOrchestrator(tick={self.current_tick})"
    