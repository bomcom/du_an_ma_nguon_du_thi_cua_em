"""
environment_pressure_engine.py

Phase 5B - Environmental Pressure Engine
Chỉ tạo và quản lý áp lực môi trường.
KHÔNG sửa fitness, KHÔNG đụng ECS, KHÔNG mutate genome.
"""
"""

Tên tệp: environment_pressure_engine.py
Giai đoạn: 5B - Động cơ Áp lực Môi trường

Bản quyền © 2026 Phạm Hồng Hải Đăng.

Mọi quyền được bảo lưu.

Tài liệu này thuộc sở hữu trí tuệ của Phạm Hồng Hải Đăng.

"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any, TYPE_CHECKING
import logging
import random
import copy

if TYPE_CHECKING:
    from .ecological_niche_manager import EcologicalNicheManager

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentalPressure:
    pressure_id: str
    target_species: str
    pressure_type: str          # resource, climate, predator, competition
    severity: float             # 0.0 ~ 1.0
    duration_ticks: int
    created_tick: int
    description: str = ""


class EnvironmentPressureEngine:
    def __init__(self):
        self._active_pressures: List[EnvironmentalPressure] = []
        self._tick_counter: int = 0

    def generate_pressures(
        self,
        species_ids: Set[str],
        niche_manager: "EcologicalNicheManager",
        environment_state: Dict[str, Any]
    ) -> List[EnvironmentalPressure]:
        self._tick_counter += 1
        self._expire_old_pressures()

        new_pressures: List[EnvironmentalPressure] = []

        for species_id in species_ids:
            # Deduplication: chỉ có 1 pressure cùng type cho 1 species
            if self._has_active_pressure(species_id, "resource") and \
               self._has_active_pressure(species_id, "climate"):
                continue

            match_score = niche_manager.calculate_adaptation_score(species_id, environment_state)
            if match_score >= 0.75:
                continue

            pressure = self._create_pressure(species_id, match_score, environment_state)
            if pressure:
                self._active_pressures.append(pressure)
                new_pressures.append(pressure)

        return new_pressures

    def _create_pressure(self, species_id: str, match_score: float, env: Dict[str, Any]) -> Optional[EnvironmentalPressure]:
        severity = round(1.0 - match_score, 3)
        pressure_type, desc = self._decide_pressure_type(env)

        return EnvironmentalPressure(
            pressure_id=f"press_{self._tick_counter}_{species_id}_{random.randint(1000,9999)}",
            target_species=species_id,
            pressure_type=pressure_type,
            severity=severity,
            duration_ticks=random.randint(80, 400),
            created_tick=self._tick_counter,
            description=desc
        )

    def _decide_pressure_type(self, env: Dict[str, Any]) -> tuple:
        # Có thể mở rộng sau bằng PressureRule system
        if len(env.get("available_resources", set())) < 3:
            return "resource", "Resource scarcity"
        temp = env.get("temperature", 20.0)
        if abs(temp - 20) > 12:
            return "climate", "Extreme climate"
        if env.get("population_density", 0) > 0.75:
            return "competition", "High competition"
        return "climate", "General environmental stress"

    def _has_active_pressure(self, species_id: str, p_type: str) -> bool:
        return any(p.target_species == species_id and p.pressure_type == p_type
                   for p in self._active_pressures)

    def _expire_old_pressures(self):
        self._active_pressures = [
            p for p in self._active_pressures
            if self._tick_counter - p.created_tick < p.duration_ticks
        ]

    def get_active_pressures(self) -> List[EnvironmentalPressure]:
        return copy.deepcopy(self._active_pressures)

    def get_pressures_for_species(self, species_id: str) -> List[EnvironmentalPressure]:
        return [copy.deepcopy(p) for p in self._active_pressures if p.target_species == species_id]

    def __repr__(self) -> str:
        return f"EnvironmentPressureEngine(active={len(self._active_pressures)}, tick={self._tick_counter})"
    