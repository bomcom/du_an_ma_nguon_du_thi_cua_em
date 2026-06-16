# -*- coding: utf-8 -*-
"""
simulation/world_field_generator.py
===================================

Dynamic Reality Field Generator

Prompt
    ↓
Matrix Solver
    ↓
World Fields
    ↓
Terrain / Climate / Resources
    ↓
Renderer

Không sinh biome cố định.

Toàn bộ thế giới được tạo từ trạng thái toán học hiện tại.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import threading
import numpy as np


# ==========================================================
# FIELD SNAPSHOT
# ==========================================================

@dataclass
class WorldFieldSnapshot:

    tick: int

    height_field: np.ndarray

    climate_field: np.ndarray

    resource_field: np.ndarray

    biome_field: np.ndarray

    entropy: float

    prey_population: float

    predator_population: float


# ==========================================================
# GENERATOR
# ==========================================================

class WorldFieldGenerator:

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        seed: int = 42
    ):

        self.width = width
        self.height = height

        self.rng = np.random.default_rng(seed)

        self._lock = threading.RLock()

        self._snapshot = None

        self._base_noise = self.rng.random(
            (height, width)
        ).astype(np.float32)

    # ======================================================
    # PUBLIC
    # ======================================================

    def update(
        self,
        math_state: Dict[str, Any],
        hardware_state: Optional[Dict[str, Any]] = None
    ) -> None:

        with self._lock:

            height_field = self._generate_height_field(
                math_state
            )

            climate_field = self._generate_climate_field(
                math_state,
                height_field
            )

            resource_field = self._generate_resource_field(
                math_state,
                climate_field
            )

            biome_field = self._generate_biome_field(
                height_field,
                climate_field,
                resource_field
            )

            self._snapshot = WorldFieldSnapshot(
                tick=int(math_state.get("tick", 0)),
                height_field=height_field,
                climate_field=climate_field,
                resource_field=resource_field,
                biome_field=biome_field,
                entropy=float(
                    math_state.get(
                        "lorenz",
                        [0.0, 0.0, 0.0]
                    )[0]
                ),
                prey_population=float(
                    math_state.get("prey", 0.0)
                ),
                predator_population=float(
                    math_state.get("predator", 0.0)
                )
            )

    # ======================================================

    def get_snapshot(
        self
    ) -> Optional[WorldFieldSnapshot]:

        with self._lock:

            return self._snapshot

    # ======================================================
    # FIELD GENERATORS
    # ======================================================

    def _generate_height_field(
        self,
        math_state: Dict[str, Any]
    ) -> np.ndarray:

        e_total = float(
            math_state.get("E_total", 0.0)
        )

        e_max = float(
            math_state.get("E_max", 1.0)
        )

        energy_ratio = np.clip(
            e_total / max(e_max, 1e-6),
            0.0,
            1.0
        )

        lorenz = math_state.get(
            "lorenz",
            [0.0, 0.0, 0.0]
        )

        chaos = abs(float(lorenz[0])) / 50.0

        field = (
            self._base_noise
            * (0.3 + energy_ratio)
            + chaos
        )

        return np.clip(field, 0.0, 1.0)

    # ======================================================

    def _generate_climate_field(
        self,
        math_state: Dict[str, Any],
        height_field: np.ndarray
    ) -> np.ndarray:

        prey = float(
            math_state.get("prey", 0.0)
        )

        predator = float(
            math_state.get("predator", 0.0)
        )

        balance = (
            prey - predator
        ) / max(prey + predator + 1.0, 1.0)

        climate = (
            1.0
            - height_field
            + balance
        )

        return np.clip(
            climate,
            0.0,
            1.0
        )

    # ======================================================

    def _generate_resource_field(
        self,
        math_state: Dict[str, Any],
        climate_field: np.ndarray
    ) -> np.ndarray:

        bio_energy = float(
            math_state.get(
                "E_biological",
                0.0
            )
        )

        total_energy = float(
            math_state.get(
                "E_total",
                1.0
            )
        )

        ratio = np.clip(
            bio_energy /
            max(total_energy, 1.0),
            0.0,
            1.0
        )

        resource = (
            climate_field
            * ratio
        )

        return np.clip(
            resource,
            0.0,
            1.0
        )

    # ======================================================

    def _generate_biome_field(
        self,
        height_field: np.ndarray,
        climate_field: np.ndarray,
        resource_field: np.ndarray
    ) -> np.ndarray:

        biome = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        water_mask = (
            height_field < 0.20
        )

        grass_mask = (
            (height_field >= 0.20)
            &
            (resource_field > 0.40)
        )

        forest_mask = (
            (resource_field > 0.70)
            &
            (climate_field > 0.50)
        )

        mountain_mask = (
            height_field > 0.85
        )

        desert_mask = (
            resource_field < 0.15
        )

        biome[water_mask] = 0
        biome[grass_mask] = 1
        biome[forest_mask] = 2
        biome[desert_mask] = 3
        biome[mountain_mask] = 4

        return biome

    # ======================================================
    # SAMPLING
    # ======================================================

    def sample_height(
        self,
        x: float,
        y: float
    ) -> float:

        snapshot = self._snapshot

        if snapshot is None:
            return 0.0

        ix = int(
            np.clip(
                x,
                0,
                self.width - 1
            )
        )

        iy = int(
            np.clip(
                y,
                0,
                self.height - 1
            )
        )

        return float(
            snapshot.height_field[
                iy,
                ix
            ]
        )

    # ======================================================

    def sample_resource(
        self,
        x: float,
        y: float
    ) -> float:

        snapshot = self._snapshot

        if snapshot is None:
            return 0.0

        ix = int(
            np.clip(
                x,
                0,
                self.width - 1
            )
        )

        iy = int(
            np.clip(
                y,
                0,
                self.height - 1
            )
        )

        return float(
            snapshot.resource_field[
                iy,
                ix
            ]
        )

    # ======================================================

    def sample_biome(
        self,
        x: float,
        y: float
    ) -> int:

        snapshot = self._snapshot

        if snapshot is None:
            return 0

        ix = int(
            np.clip(
                x,
                0,
                self.width - 1
            )
        )

        iy = int(
            np.clip(
                y,
                0,
                self.height - 1
            )
        )

        return int(
            snapshot.biome_field[
                iy,
                ix
            ]
        )
    