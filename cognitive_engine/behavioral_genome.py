"""
behavioral_genome.py

Phase 6A - Behavioral Genome Layer (Civilization-Capable Evolution Upgrade)
Part of the large-scale evolutionary simulation for MA_NGUON_DU_AN_QUOC_GIA.

This module establishes the genetic foundation for behavior (behavioral traits) 
before high-level cognitive structures (goals, tools, culture, neural networks) are formed.

Strict Boundaries:
- NO Neural Network computation (No Torch, No Weights/Nodes).
- NO direct interaction with ECSRegistry.
- NO tool creation or recipe management.
- Purely manages behavior trait allocation, inheritance, and mutation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set
import logging
import random
import copy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BehavioralTraits:
    """
    Immutable representation of foundational behavioral predisposition values.
    All traits are strictly bounded within the closed interval [0.0, 1.0].
    
    Includes both primitive biological triggers and civilization-emergence triggers.
    """
    # --- Nhóm 1: Gene Sinh Tồn Sinh Học (Primitive Survival Traits) ---
    aggression: float     # Xu hướng tấn công, cạnh tranh tài nguyên hoặc tranh giành lãnh thổ.
    curiosity: float      # Động lực khám phá môi trường, các vùng đất mới và vật thể lạ.
    fear: float           # Mức độ cảnh giác, né tránh hiểm họa hoặc bỏ chạy khi bất lợi.
    cooperation: float    # Sẵn sàng chia sẻ tài nguyên, bảo vệ đồng loại, liên kết bầy đàn.

    # --- Nhóm 2: Gene Kiến Tạo Văn Minh (Civilization-Emergence Traits) ---
    innovation: float     # Xu hướng thử nghiệm ngẫu nhiên, kết hợp các vật phẩm, phá vỡ công thức cũ.
    resourcefulness: float # Khả năng tận dụng phế liệu, tái sử dụng vật liệu thô, tối ưu hóa công cụ hiện có.
    abstraction: float     # Tư duy bắc cầu/suy luận logic (Ví dụ: A+B=C thì A+D=?); quyết định tốc độ tiếp thu học thuyết.

    def validate(self) -> bool:
        """Verify if all foundational traits reside within valid logical bounds [0.0, 1.0]."""
        for trait_name, val in [
            ("aggression", self.aggression),
            ("curiosity", self.curiosity),
            ("fear", self.fear),
            ("cooperation", self.cooperation),
            ("innovation", self.innovation),
            ("resourcefulness", self.resourcefulness),
            ("abstraction", self.abstraction)
        ]:
            if not (0.0 <= val <= 1.0):
                logger.warning(f"Trait '{trait_name}' value {val} is outside valid range [0.0, 1.0]")
                return False
        return True


@dataclass
class BehaviorGenome:
    """
    The complete behavioral genome configuration for an individual entity.
    Acts as a standalone genetic blueprint controlling innate behavioral baselines.
    """
    entity_id: int
    traits: BehavioralTraits
    generation: int = 1
    mutation_history_count: int = 0
    # Scalable field for hyper-specific traits without altering core mathematical signatures
    extended_traits: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Đảm bảo dữ liệu gene luôn được chuẩn hóa cứng ngay khi khởi tạo
        self.clamp_and_validate_all()

    def clamp_and_validate_all(self) -> None:
        """Enforces hard limits on all behavioral coordinates to prevent system drift."""
        clamped_core = BehavioralTraits(
            aggression=max(0.0, min(1.0, self.traits.aggression)),
            curiosity=max(0.0, min(1.0, self.traits.curiosity)),
            fear=max(0.0, min(1.0, self.traits.fear)),
            cooperation=max(0.0, min(1.0, self.traits.cooperation)),
            innovation=max(0.0, min(1.0, self.traits.innovation)),
            resourcefulness=max(0.0, min(1.0, self.traits.resourcefulness)),
            abstraction=max(0.0, min(1.0, self.traits.abstraction))
        )
        object.__setattr__(self, 'traits', clamped_core)

        # Tránh trôi biên dữ liệu cho các thuộc tính mở rộng
        for k, v in self.extended_traits.items():
            self.extended_traits[k] = max(0.0, min(1.0, v))


class BehaviorMutationEngine:
    """
    Handles genetic drift and mutation logic for behavioral traits.
    Operates algorithmically using localized Gaussian distribution shifts.
    """

    def __init__(self, default_mutation_rate: float = 0.05, default_mutation_strength: float = 0.08):
        self.mutation_rate = max(0.0, min(1.0, default_mutation_rate))
        self.mutation_strength = default_mutation_strength

    def mutate_genome(self, genome: BehaviorGenome, custom_rate: Optional[float] = None, custom_strength: Optional[float] = None) -> BehaviorGenome:
        """
        Applies a non-destructive structural mutation to a behavioral genome copy.
        Uses a Gaussian distribution centered at the current value to simulate natural continuous drift.
        """
        rate = custom_rate if custom_rate is not None else self.mutation_rate
        strength = custom_strength if custom_strength is not None else self.mutation_strength

        curr_traits = genome.traits
        mutated_count = genome.mutation_history_count

        def _mutate_val(val: float) -> float:
            if random.random() < rate:
                nonlocal mutated_count
                mutated_count += 1
                # Gaussian offset to maintain realistic continuous mutation steps
                delta = random.gauss(0.0, strength)
                return max(0.0, min(1.0, val + delta))
            return val

        new_traits = BehavioralTraits(
            aggression=_mutate_val(curr_traits.aggression),
            curiosity=_mutate_val(curr_traits.curiosity),
            fear=_mutate_val(curr_traits.fear),
            cooperation=_mutate_val(curr_traits.cooperation),
            innovation=_mutate_val(curr_traits.innovation),
            resourcefulness=_mutate_val(curr_traits.resourcefulness),
            abstraction=_mutate_val(curr_traits.abstraction)
        )

        new_extended = {}
        for k, v in genome.extended_traits.items():
            new_extended[k] = _mutate_val(v)

        return BehaviorGenome(
            entity_id=genome.entity_id,
            traits=new_traits,
            generation=genome.generation,
            mutation_history_count=mutated_count,
            extended_traits=new_extended
        )


class BehaviorInheritanceEngine:
    """
    Executes sexual/asexual reproduction operations over BehaviorGenomes.
    Combines parents using a blending crossover mechanism to prevent immediate trait divergence.
    """

    @staticmethod
    def crossover(parent_a: BehaviorGenome, parent_b: BehaviorGenome, child_id: int) -> BehaviorGenome:
        """
        Performs continuous blend crossover between two parental behavioral profiles.
        Formula: Child_Trait = Alpha * ParentA_Trait + (1 - Alpha) * ParentB_Trait
        """
        # Dynamic mixing coefficient per reproduction event to simulate uneven allele dominance
        alpha = random.uniform(0.1, 0.9)

        t_a = parent_a.traits
        t_b = parent_b.traits

        child_traits = BehavioralTraits(
            aggression=round(alpha * t_a.aggression + (1.0 - alpha) * t_b.aggression, 4),
            curiosity=round(alpha * t_a.curiosity + (1.0 - alpha) * t_b.curiosity, 4),
            fear=round(alpha * t_a.fear + (1.0 - alpha) * t_b.fear, 4),
            cooperation=round(alpha * t_a.cooperation + (1.0 - alpha) * t_b.cooperation, 4),
            innovation=round(alpha * t_a.innovation + (1.0 - alpha) * t_b.innovation, 4),
            resourcefulness=round(alpha * t_a.resourcefulness + (1.0 - alpha) * t_b.resourcefulness, 4),
            abstraction=round(alpha * t_a.abstraction + (1.0 - alpha) * t_b.abstraction, 4)
        )

        # Trộn các thuộc tính mở rộng (nếu có)
        child_extended = {}
        all_ext_keys = set(parent_a.extended_traits.keys()) | set(parent_b.extended_traits.keys())
        for key in all_ext_keys:
            val_a = parent_a.extended_traits.get(key, 0.5)
            val_b = parent_b.extended_traits.get(key, 0.5)
            child_extended[key] = round(alpha * val_a + (1.0 - alpha) * val_b, 4)

        max_gen = max(parent_a.generation, parent_b.generation) + 1

        return BehaviorGenome(
            entity_id=child_id,
            traits=child_traits,
            generation=max_gen,
            mutation_history_count=0,
            extended_traits=child_extended
        )

    @staticmethod
    def replicate_asexual(parent: BehaviorGenome, child_id: int) -> BehaviorGenome:
        """Generates an exact line-copy of a single parent for clonal species lineages."""
        return BehaviorGenome(
            entity_id=child_id,
            traits=copy.deepcopy(parent.traits),
            generation=parent.generation + 1,
            mutation_history_count=parent.mutation_history_count,
            extended_traits=copy.deepcopy(parent.extended_traits)
        )


class BehavioralGenomeManager:
    """
    Central tracking system for behavioral blueprints across the active simulation runtime.
    Maintains localized registry decoupled entirely from heavy ECS data structures.
    """

    def __init__(self):
        self._registry: Dict[int, BehaviorGenome] = {}
        self.mutation_engine = BehaviorMutationEngine()
        self.inheritance_engine = BehaviorInheritanceEngine()

    def register_genome(self, genome: BehaviorGenome) -> None:
        """Saves a genome profile into the dedicated behavioral registry."""
        if genome.traits.validate():
            self._registry[genome.entity_id] = genome
        else:
            raise ValueError(f"Cannot register invalid behavior genome for entity {genome.entity_id}")

    def get_genome(self, entity_id: int) -> Optional[BehaviorGenome]:
        """Fetches an existing entity's behavior genome profile."""
        return self._registry.get(entity_id)

    def remove_genome(self, entity_id: int) -> bool:
        """Removes profile from tracker upon entity death or despawn."""
        if entity_id in self._registry:
            del self._registry[entity_id]
            return True
        return False

    def get_behavior_profile_matrix(self, entity_id: int) -> Optional[Dict[str, float]]:
        """
        Exposes raw trait metrics strictly for downstream reading by Phase 6B/6D engines.
        This provides the exact 'prior bias' input needed by GoalFormationEngine.
        """
        genome = self.get_genome(entity_id)
        if not genome:
            return None
        
        profile = {
            "aggression": genome.traits.aggression,
            "curiosity": genome.traits.curiosity,
            "fear": genome.traits.fear,
            "cooperation": genome.traits.cooperation,
            "innovation": genome.traits.innovation,
            "resourcefulness": genome.traits.resourcefulness,
            "abstraction": genome.traits.abstraction,
            "generation": float(genome.generation)
        }
        for k, v in genome.extended_traits.items():
            profile[f"ext_{k}"] = v
        return profile

    def list_monitored_entities(self) -> Set[int]:
        """Returns IDs of all tracked behavioral profiles."""
        return set(self._registry.keys())

    def __repr__(self) -> str:
        return f"BehavioralGenomeManager(tracked_profiles={len(self._registry)})"


# =============== Khối Kiểm Tra Thực Thi Thiết Kế ===============
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Testing Stage 6A: Civilization-Capable Behavioral Genome ===")
    
    manager = BehavioralGenomeManager()

    # 1. Khởi tạo Archetype 1: Chủng tộc có tiềm năng xây dựng nền văn minh (Civilization Archetype)
    # Đặc trưng: Thấp về hung dữ, cực cao về đổi mới, tư duy trừu tượng và tối ưu hóa tài nguyên.
    civ_traits = BehavioralTraits(
        aggression=0.20, curiosity=0.85, fear=0.30, cooperation=0.75,
        innovation=0.90, resourcefulness=0.80, abstraction=0.85
    )
    entity_civ = BehaviorGenome(entity_id=701, traits=civ_traits, extended_traits={"mana_sensitivity": 0.60})

    # Khởi tạo Archetype 2: Chủng tộc dã thú nguyên thủy (Primitive Beast Archetype)
    # Đặc trưng: Hung dữ cao, sợ hãi cao, nhưng đổi mới và trừu tượng gần như bằng không.
    beast_traits = BehavioralTraits(
        aggression=0.90, curiosity=0.30, fear=0.60, cooperation=0.20,
        innovation=0.05, resourcefulness=0.10, abstraction=0.02
    )
    entity_beast = BehaviorGenome(entity_id=702, traits=beast_traits)

    manager.register_genome(entity_civ)
    manager.register_genome(entity_beast)
    print(f"Khởi tạo thành công hệ thống quản lý Gene: {manager}")

    # 2. Kiểm tra lai ghép (Crossover) giữa loài tiến hóa cao và loài dã thú
    child_id = 801
    child_genome = manager.inheritance_engine.crossover(entity_civ, entity_beast, child_id=child_id)
    manager.register_genome(child_genome)
    
    print(f"\n--- Kết Quả Lai Ghép Thế Hệ F1 (ID: {child_id}) ---")
    matrix_f1 = manager.get_behavior_profile_matrix(child_id)
    if matrix_f1:
        for trait, val in matrix_f1.items():
            print(f"  > {trait}: {val}")

    # 3. Kiểm tra Đột biến di truyền (Mutation Engine) tác động lên các gene văn minh mới bổ sung
    print("\n--- Kiểm Tra Đột Biến Ép Buộc (Tỷ lệ 100%) ---")
    mutated_child = manager.mutation_engine.mutate_genome(child_genome, custom_rate=1.0, custom_strength=0.15)
    manager.register_genome(mutated_child)
    
    matrix_mutated = manager.get_behavior_profile_matrix(child_id)
    if matrix_f1 and matrix_mutated:
        print(f"Chỉ số Đột Biến Trừu Tượng (Abstraction) cũ: {matrix_f1['abstraction']} -> Mới: {matrix_mutated['abstraction']}")
    print(f"Tổng số sự kiện đột biến tích lũy trong chuỗi gene: {mutated_child.mutation_history_count}")
    
    print("\nStage 6A isolated code update: PASSED.")
    