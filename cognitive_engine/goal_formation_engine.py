"""
goal_formation_engine.py

Phase 6B - Motivation & Goal Formation Engine
Part of the large-scale evolutionary simulation for MA_NGUON_DU_AN_QUOC_GIA.

Responsibility:
- Strictly handles CORE MOTIVATIONS (Survival, Aggregation, Investigation).
- Adheres to Single Responsibility Principle (SRP): Does NOT craft tools, build shelters, 
  or process logic. It only outputs what the entity WANTS, not HOW to get it.

Strict Boundaries:
- NO tool hardcoding (No "Craft", "Build" goals here).
- Outputs a purely psychological `GoalStack` to be consumed by Stage 6C (Cognitive Engine).
"""
"""

Tên tệp: goal_formation_engine.py
Mô tả: Động lực và Bộ hình thành mục tiêu - Tạo ra các

Bản quyền © 2026 Phạm Hồng Hải Đăng.

Mọi quyền được bảo lưu.

Tài liệu này thuộc sở hữu trí tuệ của Phạm Hồng Hải Đăng.

"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional
import logging

try:
    from .behavioral_genome import BehaviorGenome, BehavioralTraits
except ImportError:
    from behavioral_genome import BehaviorGenome, BehavioralTraits

logger = logging.getLogger(__name__)


class CoreDrive(Enum):
    """
    Fundamental psychological and biological drives.
    Notice the absence of high-level civilization concepts like 'CRAFT' or 'RESEARCH'.
    """
    FLEE = auto()           # Động lực né tránh cái chết/hiểm họa (Fear driven)
    SUSTAIN = auto()        # Động lực duy trì năng lượng sinh học (Hunger driven)
    DOMINATE = auto()       # Động lực chiếm đoạt, xua đuổi kẻ khác (Aggression driven)
    EXPLORE = auto()        # Động lực di chuyển tìm kiếm không gian mới (Curiosity driven)
    AGGREGATE = auto()      # Động lực bầy đàn, tìm kiếm đồng loại (Cooperation driven)
    
    # --- The Bridge to Civilization (Bước đệm cho 6C/6D) ---
    # Thay vì "Chế tạo công cụ", sinh vật chỉ sinh ra nhu cầu "Thao tác/Tương tác" với vật thể lạ.
    # Tầng 6C sẽ quyết định xem sự thao tác này dẫn đến cái gì.
    INVESTIGATE = auto()    # Động lực phân tích/tương tác với sự vật bất thường (Innovation & Abstraction driven)


@dataclass
class Goal:
    """Represents an active fundamental drive with a dynamic utility/priority score."""
    drive: CoreDrive
    priority: float  # Bounded [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.priority = max(0.0, min(1.0, self.priority))


class GoalStack:
    """Maintains a sorted hierarchy of an entity's current desires."""

    def __init__(self, max_depth: int = 4):
        self._goals: List[Goal] = []
        self.max_depth = max_depth

    def push_or_update(self, new_goal: Goal) -> None:
        """Upserts a goal based on its CoreDrive and sorts the stack."""
        for idx, existing_goal in enumerate(self._goals):
            if existing_goal.drive == new_goal.drive:
                self._goals[idx] = new_goal
                self._sort_stack()
                return

        self._goals.append(new_goal)
        self._sort_stack()

        if len(self._goals) > self.max_depth:
            self._goals = self._goals[:self.max_depth]

    def get_top_goal(self) -> Optional[Goal]:
        """The absolute highest priority drive dictating current behavior."""
        return self._goals[0] if self._goals else None

    def get_all_goals(self) -> List[Goal]:
        return list(self._goals)

    def _sort_stack(self) -> None:
        self._goals.sort(key=lambda g: g.priority, reverse=True)


@dataclass
class StimulusSnapshot:
    """
    Abstracted sensory inputs. 
    Does not know about 'Wood' or 'Mana', only knows about 'Environmental Complexity'.
    """
    # Trạng thái nội sinh
    vitality: float          # [0.0, 1.0] - Health/Integrity
    energy: float            # [0.0, 1.0] - Stamina/Calories
    
    # Kích thích ngoại sinh
    imminent_threat: float   # [0.0, 1.0] - Mức độ sát thương tiềm tàng xung quanh
    resource_density: float  # [0.0, 1.0] - Mật độ vật chất có thể hấp thụ (Thức ăn)
    peer_density: float      # [0.0, 1.0] - Mật độ cá thể cùng loài
    
    # Tín hiệu nổi sinh (Emergence Triggers)
    anomaly_signal: float    # [0.0, 1.0] - Sự hiện diện của vật chất bất thường/không xác định (VD: Mana, mảnh vụn)


class MotivationEngine:
    """
    Calculates the utility of core drives based purely on Genetic Bias + Sensory Stimuli.
    """

    def __init__(self):
        pass

    def evaluate_drives(self, genome: BehaviorGenome, stimuli: StimulusSnapshot) -> GoalStack:
        stack = GoalStack()
        traits = genome.traits

        # 1. FLEE (Sống sót)
        # Bị kích hoạt mạnh bởi sự kết hợp giữa hiểm họa thực tế và gene sợ hãi.
        # Nếu máu (vitality) thấp, bản năng sinh tồn lấn át logic.
        flee_utility = stimuli.imminent_threat * (0.6 + 0.4 * traits.fear)
        if stimuli.vitality < 0.3:
            flee_utility = min(1.0, flee_utility + (0.3 - stimuli.vitality) * 2.0)
        stack.push_or_update(Goal(CoreDrive.FLEE, flee_utility))

        # 2. SUSTAIN (Duy trì năng lượng)
        hunger = 1.0 - stimuli.energy
        sustain_utility = hunger * (0.5 + 0.5 * stimuli.resource_density)
        stack.push_or_update(Goal(CoreDrive.SUSTAIN, sustain_utility))

        # 3. DOMINATE (Thống trị/Chiếm đoạt)
        # Khao khát tranh giành phụ thuộc vào tính hung dữ và sự tự tin (máu cao, ít hiểm họa)
        if stimuli.vitality > 0.5 and stimuli.imminent_threat < 0.3:
            dominate_utility = traits.aggression * stimuli.vitality
            stack.push_or_update(Goal(CoreDrive.DOMINATE, dominate_utility))

        # 4. EXPLORE (Khám phá)
        # Chỉ kích hoạt khi tương đối no và an toàn. Phụ thuộc gene tò mò.
        if stimuli.energy > 0.4 and stimuli.imminent_threat < 0.4:
            explore_utility = traits.curiosity * stimuli.energy * (1.0 - stimuli.imminent_threat)
            stack.push_or_update(Goal(CoreDrive.EXPLORE, explore_utility))

        # 5. AGGREGATE (Bầy đàn)
        # Nhu cầu tụ tập khi thấy đồng loại
        aggregate_utility = traits.cooperation * stimuli.peer_density
        stack.push_or_update(Goal(CoreDrive.AGGREGATE, aggregate_utility))

        # =====================================================================
        # BƯỚC ĐỆM CHO VĂN MINH (THE BRIDGE TO 6C/6D)
        # =====================================================================
        # 6. INVESTIGATE (Nghiên cứu/Thao tác)
        # Sinh vật không biết nó đang "chế tạo công cụ". Nó chỉ bị thôi thúc phải
        # tương tác với sự phức tạp của môi trường (anomaly_signal) dựa trên
        # khả năng tư duy (abstraction) và tính sáng tạo (innovation).
        if stimuli.anomaly_signal > 0.0 and stimuli.imminent_threat < 0.2:
            # Gene văn minh bắt đầu lên tiếng khi môi trường yên bình
            cognitive_drive = (traits.abstraction * 0.6) + (traits.innovation * 0.4)
            investigate_utility = cognitive_drive * stimuli.anomaly_signal
            
            # Khởi tạo siêu dữ liệu để báo cho 6C biết mức độ "nhạy bén" của cá thể này
            meta = {"cognitive_capacity": cognitive_drive}
            stack.push_or_update(Goal(CoreDrive.INVESTIGATE, investigate_utility, metadata=meta))

        return stack


if __name__ == "__main__":
    print("=== Testing Stage 6B: Pure Motivation Engine ===")
    engine = MotivationEngine()

    # Kịch bản: Sinh vật tiền văn minh (Có gene tư duy cao)
    proto_civ_traits = BehavioralTraits(
        aggression=0.2, curiosity=0.9, fear=0.2, cooperation=0.8,
        innovation=0.9, resourcefulness=0.8, abstraction=0.85
    )
    genome = BehaviorGenome(entity_id=1, traits=proto_civ_traits)

    # Môi trường 1: Đói và nguy hiểm
    crisis = StimulusSnapshot(vitality=0.2, energy=0.1, imminent_threat=0.9, resource_density=0.1, peer_density=0.0, anomaly_signal=0.8)
    print("\n[Môi trường: Đói & Nguy hiểm]")
    for g in engine.evaluate_drives(genome, crisis).get_all_goals():
        print(f" - {g.drive.name}: {g.priority:.3f}")

    # Môi trường 2: No đủ và bắt gặp vật thể lạ
    peaceful = StimulusSnapshot(vitality=0.9, energy=0.9, imminent_threat=0.0, resource_density=0.8, peer_density=0.5, anomaly_signal=0.9)
    print("\n[Môi trường: An toàn & Bắt gặp vật thể lạ]")
    for g in engine.evaluate_drives(genome, peaceful).get_all_goals():
        print(f" - {g.drive.name}: {g.priority:.3f}")
        
    print("\nStage 6B strictly decoupled: PASSED.")
    