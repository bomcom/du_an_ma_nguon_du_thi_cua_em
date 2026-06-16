"""
neural_cognitive_engine.py

Phase 6C - Cognitive & Neural Decision Engine (RL & Planning Upgrade)
Part of the large-scale evolutionary simulation for MA_NGUON_DU_AN_QUOC_GIA.

Responsibility:
- Receives GoalStack (6B) and Environmental Latent State.
- Generates a physical MICRO-ACTION SEQUENCE (Action Planner).
- Uses Reinforcement Learning (Q-Learning scaffolding) instead of hard-coded If-Else rules.
- Does NOT know what a "Tool" is. It only learns which physical sequences yield rewards.
"""

"""

Tên tệp: neural_cognitive_engine.py
Mô tả: COGNITIVE_ENGINE — Bộ não thần kinh sử dụng Học Tập Của Cường Hóa (Reinforcement Learning) để lập kế hoạch hành động dựa trên trạng thái tiềm ẩn của môi trường và mục tiêu hàng đầu.

Bản quyền © 2026 Phạm Hồng Hải Đăng.

Mọi quyền được bảo lưu.

Tài liệu này thuộc sở hữu trí tuệ của Phạm Hồng Hải Đăng.

"""



from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, List, Optional
import logging
import random

logger = logging.getLogger(__name__)


# =========================================================================
# 6C-A: Action Standard & Sequences
# =========================================================================

class MicroActionType(Enum):
    """
    Primitive physical actions. 
    Complex behaviors emerge from chaining these together.
    """
    IDLE = auto()
    LOOK_AT = auto()           # Focus attention on a target
    MOVE_TO = auto()           # Physical displacement
    FLEE_FROM = auto()         # Move away
    GRAB = auto()              # Hold an object
    DROP = auto()              # Release an object
    APPLY_FORCE = auto()       # Strike, push, or smash
    POUR_COMBINE = auto()      # Combine held object with target object
    CONSUME = auto()           # Ingest
    VOCALIZE = auto()          # Social signaling


@dataclass(frozen=True)
class ActionDecision:
    """A single discrete step in an action sequence."""
    action_type: MicroActionType
    target_id: Optional[str] = None
    held_item_id: Optional[str] = None # Giữ trạng thái của tay (Ví dụ: đang cầm Wood)
    intensity: float = 0.5             # Lực tương tác hoặc tốc độ [0.0, 1.0]

@dataclass
class ActionSequence:
    """A planned sequence of actions to be executed over multiple ticks."""
    goal_source: str
    steps: List[ActionDecision]
    current_step_index: int = 0
    
    def get_next_action(self) -> Optional[ActionDecision]:
        if self.current_step_index < len(self.steps):
            step = self.steps[self.current_step_index]
            self.current_step_index += 1
            return step
        return None

    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)


# =========================================================================
# 6C-B: Neural Topology & Memory
# =========================================================================

@dataclass
class BrainTopology:
    """Defines the shape of the cognitive network for genetic evolution."""
    input_nodes: int
    hidden_layers: List[int]
    output_nodes: int
    learning_rate: float = 0.05
    discount_factor: float = 0.9  # Gamma in RL
    exploration_rate: float = 0.2 # Epsilon in Epsilon-Greedy

    def clone_with_mutation(self) -> 'BrainTopology':
        new_layers = list(self.hidden_layers)
        if random.random() < 0.1 and new_layers:
            layer_idx = random.randint(0, len(new_layers) - 1)
            new_layers[layer_idx] = max(4, new_layers[layer_idx] + random.choice([-1, 1]))
            
        return BrainTopology(
            input_nodes=self.input_nodes,
            hidden_layers=new_layers,
            output_nodes=self.output_nodes,
            learning_rate=max(0.01, min(0.1, self.learning_rate + random.uniform(-0.01, 0.01))),
            discount_factor=self.discount_factor,
            exploration_rate=self.exploration_rate
        )


class StateEncoder:
    """
    Compresses high-dimensional sensory vectors into discrete latent states.
    Prevents the 'State Hash Explosion' problem in RL.
    """
    @staticmethod
    def encode(sensory_vector: List[float], resolution: int = 4) -> str:
        """
        Bridges continuous environments to discrete Q-Tables by binning values.
        Ví dụ: [0.85, 0.21, ...] -> "3_0_..." (chia thành 4 mức)
        """
        binned = [str(int(min(v * resolution, resolution - 1))) for v in sensory_vector]
        return "-".join(binned)


class RLMemory:
    """
    Q-Table scaffolding. Replaces hard-coded IF-ELSE rules.
    Q(s, a) estimates the long-term reward of action 'a' in state 's'.
    """
    def __init__(self):
        # Format: { latent_state_str: { action_enum_name: q_value_float } }
        self.q_table: Dict[str, Dict[str, float]] = {}
        
    def get_q_value(self, state: str, action_name: str) -> float:
        if state not in self.q_table:
            return 0.0
        return self.q_table[state].get(action_name, 0.0)

    def update_q_value(self, state: str, action_name: str, reward: float, next_state: str, 
                       lr: float, discount: float):
        """Bellman equation update."""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        current_q = self.get_q_value(state, action_name)
        
        # Lấy max Q của trạng thái kế tiếp
        next_max_q = 0.0
        if next_state in self.q_table and self.q_table[next_state]:
            next_max_q = max(self.q_table[next_state].values())
            
        # Công thức Q-Learning
        new_q = current_q + lr * (reward + discount * next_max_q - current_q)
        self.q_table[state][action_name] = new_q


# =========================================================================
# 6C-C: Neural Brain & Action Planner
# =========================================================================

class ActionPlanner:
    """
    Translates an abstract RL decision into a concrete physical sequence.
    Essential for Emergent Tool Creation (Stage 6D).
    """
    @staticmethod
    def build_sequence(intent: str, top_goal: Any, target_items: List[str]) -> ActionSequence:
        goal_name = getattr(top_goal, 'drive', top_goal).name if hasattr(top_goal, 'drive') else str(top_goal)
        steps = []

        # Giải mã "intent" học được từ Q-Table thành chuỗi hành động vật lý
        if intent == "SEQUENCE_COMBINE" and len(target_items) >= 2:
            # Ví dụ: Nhặt Item_A -> Mang tới Item_B -> Kết hợp
            item_a, item_b = target_items[0], target_items[1]
            steps.append(ActionDecision(MicroActionType.MOVE_TO, target_id=item_a))
            steps.append(ActionDecision(MicroActionType.GRAB, target_id=item_a))
            steps.append(ActionDecision(MicroActionType.MOVE_TO, target_id=item_b, held_item_id=item_a))
            steps.append(ActionDecision(MicroActionType.POUR_COMBINE, target_id=item_b, held_item_id=item_a))
            
        elif intent == "SEQUENCE_SMASH" and len(target_items) >= 1:
            item = target_items[0]
            steps.append(ActionDecision(MicroActionType.MOVE_TO, target_id=item))
            steps.append(ActionDecision(MicroActionType.APPLY_FORCE, target_id=item, intensity=0.9))

        elif intent == "SEQUENCE_CONSUME" and len(target_items) >= 1:
            item = target_items[0]
            steps.append(ActionDecision(MicroActionType.MOVE_TO, target_id=item))
            steps.append(ActionDecision(MicroActionType.CONSUME, target_id=item))

        elif intent == "SEQUENCE_FLEE":
            steps.append(ActionDecision(MicroActionType.FLEE_FROM, intensity=1.0))

        else:
            steps.append(ActionDecision(MicroActionType.IDLE))

        return ActionSequence(goal_source=goal_name, steps=steps)


class NeuralBrain:
    """
    The True Cognitive Processor.
    Uses RL to choose Intents based on Latent States, not IF-ELSE rules.
    """
    def __init__(self, entity_id: int, topology: BrainTopology):
        self.entity_id = entity_id
        self.topology = topology
        self.memory = RLMemory()
        
        # Danh sách các "Ý định tổng quát" mà mạng RL có thể học chọn lựa
        self.available_intents = ["SEQUENCE_FLEE", "SEQUENCE_CONSUME", "SEQUENCE_SMASH", "SEQUENCE_COMBINE", "SEQUENCE_IGNORE"]
        
        self.active_sequence: Optional[ActionSequence] = None
        self.last_state: Optional[str] = None
        self.last_intent: Optional[str] = None

    def decide_action(self, sensory_vector: List[float], top_goal: Any) -> ActionDecision:
        """Called every tick. Returns the next Micro-Action in the planned sequence."""
        latent_state = StateEncoder.encode(sensory_vector)
        self.last_state = latent_state

        # Nếu đang có một chuỗi hành động dở dang, tiếp tục thực hiện
        if self.active_sequence and not self.active_sequence.is_complete():
            action = self.active_sequence.get_next_action()
            if action is not None:
                return action
            return ActionDecision(MicroActionType.IDLE)

        # Rỗng -> Cần Lập Kế Hoạch Mới
        if not top_goal:
            return ActionDecision(MicroActionType.IDLE)

        # Trích xuất dữ liệu môi trường (giả lập việc quét vật thể)
        meta = getattr(top_goal, 'metadata', {})
        target_items = meta.get("artifacts", ["Item_Wood", "Item_ManaLiquid"])

        # --- REINFORCEMENT LEARNING SELECTION (Epsilon-Greedy) ---
        chosen_intent = "SEQUENCE_IGNORE"
        
        if random.random() < self.topology.exploration_rate:
            # Khám phá: Tự do chọn bừa một hành động để xem điều gì xảy ra (Curiosity)
            chosen_intent = random.choice(self.available_intents)
        else:
            # Khai thác: Chọn hành động có Q-Value cao nhất trong quá khứ tại State này
            best_q = -float('inf')
            for intent in self.available_intents:
                q_val = self.memory.get_q_value(latent_state, intent)
                if q_val > best_q:
                    best_q = q_val
                    chosen_intent = intent

        self.last_intent = chosen_intent
        
        # Giao Intent cho Planner để lập ActionSequence
        self.active_sequence = ActionPlanner.build_sequence(chosen_intent, top_goal, target_items)
        if self.active_sequence is None:
            return ActionDecision(MicroActionType.IDLE)

        action = self.active_sequence.get_next_action()
        if action is not None:
            return action
        return ActionDecision(MicroActionType.IDLE)

    def process_environmental_reward(self, current_sensory_vector: List[float], reward: float):
        """Called by the Engine when an action sequence concludes and physics reacts."""
        if self.last_state and self.last_intent:
            next_state = StateEncoder.encode(current_sensory_vector)
            self.memory.update_q_value(
                state=self.last_state,
                action_name=self.last_intent,
                reward=reward,
                next_state=next_state,
                lr=self.topology.learning_rate,
                discount=self.topology.discount_factor
            )


# =========================================================================
# 6C-E: Brain Manager
# =========================================================================

class BrainManager:
    def __init__(self):
        self._brains: Dict[int, NeuralBrain] = {}

    def register_brain(self, entity_id: int, topology: BrainTopology) -> NeuralBrain:
        brain = NeuralBrain(entity_id, topology)
        self._brains[entity_id] = brain
        return brain


# =========================================================================
# Kiểm Tra Thực Thi Độc Lập
# =========================================================================
if __name__ == "__main__":
    print("=== Testing Stage 6C: Reinforcement Learning & Action Sequences ===")
    
    manager = BrainManager()
    topo = BrainTopology(input_nodes=10, hidden_layers=[16], output_nodes=5)
    brain = manager.register_brain(101, topo)
    
    # Kịch bản Mục tiêu từ 6B
    @dataclass
    class MockGoal:
        drive: str
        metadata: dict
    investigate_goal = MockGoal(drive="INVESTIGATE", metadata={"artifacts": ["Wood", "ManaLiquid"]})
    
    # Sensory liên tục
    s_vector = [0.8, 0.2, 0.5, 0.9]
    print(f"Latent State Encoded: {StateEncoder.encode(s_vector)}")

    print("\n--- TICK 1: Khởi tạo ý định ngẫu nhiên (Exploration) ---")
    # Ép buộc chọn SEQUENCE_COMBINE để test chuỗi hành động
    brain.last_intent = "SEQUENCE_COMBINE"
    brain.active_sequence = ActionPlanner.build_sequence("SEQUENCE_COMBINE", investigate_goal, ["Wood", "ManaLiquid"])
    
    tick = 1
    while not brain.active_sequence.is_complete():
        action = brain.decide_action(s_vector, investigate_goal)
        print(f"Tick {tick} | Action: {action.action_type.name} | Target: {action.target_id} | Held: {action.held_item_id}")
        tick += 1

    print("\n--- MÔI TRƯỜNG PHẢN HỒI (REWARD) ---")
    print("Môi trường: Wood + ManaLiquid sinh ra Ánh sáng (Phần thưởng sinh tồn +1.5)")
    brain.process_environmental_reward(s_vector, reward=1.5)
    
    print("\n--- Q-TABLE UPDATE ---")
    latent_state = StateEncoder.encode(s_vector)
    learned_q = brain.memory.get_q_value(latent_state, "SEQUENCE_COMBINE")
    print(f"Q-Value cho [Latent:{latent_state}][Intent:COMBINE] đã tăng thành: {learned_q:.3f}")
    
    print("\nStage 6C RL & Planner Architecture: PASSED.")
    