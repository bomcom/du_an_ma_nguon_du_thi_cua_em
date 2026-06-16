"""
ecological_niche_manager.py

Phase 5A - Ecological Niche Manager
Part of the large-scale evolutionary simulation for MA_NGUON_DU_AN_QUOC_GIA.

This module is responsible for:
- Storing species ecological preferences
- Evaluating how well a species matches current environment
- Providing data for EnvironmentPressureEngine (Phase 5B)

Strict boundaries:
- Does NOT touch ECSRegistry, Genome, Fitness, or Lineage
- Pure ecological reasoning layer
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Set, Tuple, Optional, Any
import logging
import copy
import threading

logger = logging.getLogger(__name__)


@dataclass
class SpeciesNiche:
    """Ecological niche definition for a species."""
    species_id: str

    # Preferred ECS component types (e.g. "ColdResistance", "ManaAffinity")
    preferred_components: Set[str] = field(default_factory=set)

    # Preferred resources (e.g. "meat", "mana_crystals", "grass")
    preferred_resources: Set[str] = field(default_factory=set)

    # Temperature range (°C)
    preferred_temperature_range: Tuple[float, float] = (0.0, 30.0)

    # Humidity range [0.0 - 1.0]
    preferred_humidity_range: Tuple[float, float] = (0.2, 0.8)

    # Terrain types
    terrain_preferences: Set[str] = field(default_factory=set)

    description: str = ""


class EcologicalNicheManager:
    """
    Manages species ecological niches.
    One instance per simulation world (managed by orchestrator).
    """

    def __init__(self):
        self._niches: Dict[str, SpeciesNiche] = {}
        self._data_lock = threading.Lock()

    def _validate_niche(self, niche: SpeciesNiche) -> bool:
        """Validate niche data integrity."""
        if not niche.species_id or not isinstance(niche.species_id, str):
            logger.warning("Invalid species_id")
            return False

        t_min, t_max = niche.preferred_temperature_range
        if t_min > t_max:
            logger.warning(f"Temperature range invalid for {niche.species_id}: {t_min} > {t_max}")
            return False

        h_min, h_max = niche.preferred_humidity_range
        if not (0.0 <= h_min <= h_max <= 1.0):
            logger.warning(f"Humidity range invalid for {niche.species_id}: {h_min}-{h_max}")
            return False

        return True

    def register_species_niche(self, niche: SpeciesNiche) -> bool:
        """Register or update a species niche."""
        if not self._validate_niche(niche):
            return False

        with self._data_lock:
            self._niches[niche.species_id] = copy.deepcopy(niche)
            logger.info(f"Registered/Updated niche for species: {niche.species_id}")
            return True

    def update_species_niche(self, species_id: str, **updates) -> bool:
        """Partially update an existing niche."""
        with self._data_lock:
            if species_id not in self._niches:
                logger.warning(f"Species not found: {species_id}")
                return False

            old_niche = self._niches[species_id]
            new_niche = replace(old_niche, **updates)

            if not self._validate_niche(new_niche):
                return False

            self._niches[species_id] = new_niche
            logger.info(f"Updated niche for species: {species_id}")
            return True

    def get_species_niche(self, species_id: str) -> Optional[SpeciesNiche]:
        """Return a deep copy of the niche."""
        with self._data_lock:
            niche = self._niches.get(species_id)
            return copy.deepcopy(niche) if niche else None

    def get_species_environment_profile(self, species_id: str) -> Optional[Dict[str, Any]]:
        """Return a flat profile for easy consumption by 5B."""
        niche = self.get_species_niche(species_id)
        if not niche:
            return None
        return {
            "species_id": niche.species_id,
            "preferred_resources": list(niche.preferred_resources),
            "preferred_components": list(niche.preferred_components),
            "temp_range": niche.preferred_temperature_range,
            "humidity_range": niche.preferred_humidity_range,
            "terrain": list(niche.terrain_preferences),
            "description": niche.description
        }

    def calculate_environment_match(
        self,
        species_id: str,
        current_temperature: float,
        current_humidity: float,
        available_resources: Set[str],
        current_terrain: str
    ) -> float:
        """
        Calculate how well a species matches current environment.
        Returns score in range [0.0 - 1.0]
        """
        niche = self.get_species_niche(species_id)
        if not niche:
            return 0.0

        score = 1.0

        # Temperature match
        t_min, t_max = niche.preferred_temperature_range
        if current_temperature < t_min:
            score -= 0.4 * (t_min - current_temperature) / max(1.0, abs(t_min))
        elif current_temperature > t_max:
            score -= 0.4 * (current_temperature - t_max) / max(1.0, abs(t_max))

        # Humidity match
        h_min, h_max = niche.preferred_humidity_range
        if current_humidity < h_min:
            score -= 0.3 * (h_min - current_humidity)
        elif current_humidity > h_max:
            score -= 0.3 * (current_humidity - h_max)

        # Resource compatibility
        if niche.preferred_resources:
            overlap = len(niche.preferred_resources & available_resources)
            resource_score = overlap / len(niche.preferred_resources) if niche.preferred_resources else 0.0
            score -= 0.2 * (1.0 - resource_score)

        # Terrain match
        if niche.terrain_preferences and current_terrain not in niche.terrain_preferences:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def calculate_adaptation_score(self, species_id: str, environment_state: Dict[str, Any]) -> float:
        """Higher-level convenience method using environment dict."""
        return self.calculate_environment_match(
            species_id=species_id,
            current_temperature=environment_state.get("temperature", 20.0),
            current_humidity=environment_state.get("humidity", 0.5),
            available_resources=environment_state.get("available_resources", set()),
            current_terrain=environment_state.get("terrain", "")
        )

    def list_species(self) -> Set[str]:
        with self._data_lock:
            return set(self._niches.keys())

    def get_all_niches(self) -> Dict[str, SpeciesNiche]:
        """Return deep snapshot to prevent external mutation."""
        with self._data_lock:
            return copy.deepcopy(self._niches)

    def remove_species_niche(self, species_id: str) -> bool:
        with self._data_lock:
            if species_id in self._niches:
                del self._niches[species_id]
                logger.info(f"Removed niche for species: {species_id}")
                return True
            return False

    def __repr__(self) -> str:
        with self._data_lock:
            return f"EcologicalNicheManager(species_count={len(self._niches)})"


# =============== Demo / Test Data ===============
def initialize_demo_niches(manager: EcologicalNicheManager):
    """Initialize example niches."""
    wolf = SpeciesNiche(
        species_id="wolf",
        preferred_components={"Carnivore", "PackHunter"},
        preferred_resources={"meat", "small_game"},
        preferred_temperature_range=(-15.0, 25.0),
        preferred_humidity_range=(0.3, 0.7),
        terrain_preferences={"forest", "tundra", "plains"},
        description="Temperate apex predator"
    )

    crystal_wolf = SpeciesNiche(
        species_id="crystal_wolf",
        preferred_components={"ManaAffinity", "CrystalResonance"},
        preferred_resources={"mana_crystals", "ethereal_energy"},
        preferred_temperature_range=(-5.0, 12.0),
        preferred_humidity_range=(0.4, 0.65),
        terrain_preferences={"crystal_forest", "ice_caverns"},
        description="Magical crystal-attuned predator"
    )

    manager.register_species_niche(wolf)
    manager.register_species_niche(crystal_wolf)
    logger.info("Demo ecological niches initialized.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = EcologicalNicheManager()
    initialize_demo_niches(manager)

    print(manager)
    print("Wolf match in forest/cold:", manager.calculate_environment_match(
        "wolf", -5.0, 0.6, {"meat"}, "forest"
    ))
    print("Crystal wolf match in desert:", manager.calculate_environment_match(
        "crystal_wolf", 35.0, 0.1, {"sand"}, "desert"
    ))
    