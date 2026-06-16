"""
evolution/genetic_evolution.py
==============================
Deep Machine Learning & Matrix Topology Engine (Evolutionary Agent Core)

Architecture
------------
    Vận hành các Mạng nơ-ron kích thước siêu nhỏ (Micro-Neural Networks) cho các tác nhân NPC.
    Học máy diễn ra thông qua Thuật toán Di truyền (Genetic Algorithm) dưới áp lực chọn lọc
    từ các tập hợp vật lý vĩ mô S_1 -> S_n.

    Neural Network Structure (Per Agent):
        - Input Layer: [x, y, dx_target, dy_target, current_energy, target_density]
        - Hidden Layer: 12 neurons (Tanh activation)
        - Output Layer: [vx, vy] (Vectơ vận tốc phản xạ)

    Genetic Flow:
        1. Forward Pass (Phản xạ sinh tồn): Biến ma trận môi trường thành hành vi cơ học.
        2. Fitness Evaluation: Đánh giá độ thích nghi dựa trên năng lượng sống sót.
        3. Selection & Crossover: Chọn lọc tinh hoa và lai ghép ma trận trọng số (Weights W, Biases b).
        4. Mutation: Đột biến xác suất ngẫu nhiên để tìm hướng tiến hóa mới.
"""

import logging
import random
from typing import Dict, List, Set, Tuple, Any

import numpy as np

# Thử sử dụng CuPy nếu đã được cấu hình từ Lõi Toán, nếu không dùng NumPy
try:
    import cupy as cp  # type: ignore[import-not-found]
    _CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

logger = logging.getLogger("EvolutionaryCore")

def _xp() -> Any:
    return cp if _CUPY_AVAILABLE else np

class BrainTopology:
    """Cấu trúc ma trận thần kinh của một NPC độc lập."""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 12, output_dim: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Trọng số W và Độ lệch b
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def clone(self) -> 'BrainTopology':
        """Tạo bản sao sâu của bộ não."""
        new_brain = BrainTopology(self.input_dim, self.hidden_dim, self.output_dim)
        new_brain.W1 = self.W1.copy()
        new_brain.b1 = self.b1.copy()
        new_brain.W2 = self.W2.copy()
        new_brain.b2 = self.b2.copy()
        return new_brain

class GeneticEvolutionEngine:
    """
    Động cơ Tiến hóa Di truyền.
    Tương tác trực tiếp với ECSRegistry để điều hướng hành vi NPC và sàng lọc tự nhiên.
    """
    def __init__(self, mutation_rate: float = 0.05, mutation_strength: float = 0.2):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        
        # Lưu trữ bộ não của từng Entity: entity_id -> BrainTopology
        self.brains: Dict[int, BrainTopology] = {}
        self.current_generation: int = 1
        
        logger.info(f"[EvolutionEngine] Đã khởi tạo. Tỷ lệ đột biến: {mutation_rate * 100}%")

    # =====================================================================
    # 1. QUY TRÌNH PHẢN XẠ THỜI GIAN THỰC (Forward Pass / Runtime)
    # =====================================================================

    def register_or_get_brain(self, entity_id: int) -> BrainTopology:
        """Lấy bộ não của NPC, nếu chưa có thì khởi tạo ngẫu nhiên."""
        if entity_id not in self.brains:
            self.brains[entity_id] = BrainTopology()
        return self.brains[entity_id]

    def physics_reflex_system(self, registry: Any, entities: Set[int], dt: float) -> None:
        """
        Hàm System nhúng vào ECS.
        Biến đổi thông số môi trường thành vectơ vận tốc thông qua Mạng Nơ-ron.
        Cần Component: Transform, Velocity, Energy, NeuralBrain
        """
        # Giả lập mục tiêu vĩ mô (Tọa độ lý tưởng có Mật độ Năng lượng cao nhất)
        # Trong hệ thống thực, tọa độ này sẽ được tính từ gradient của MatrixSolver
        target_x, target_y = 500.0, 500.0 
        target_density = 100.0

        for eid in entities:
            # Bỏ qua nếu Entity đang bị người dùng "Nhập hồn" (Soul Possession)
            brain_comp = registry.get_component_snapshot(eid, "NeuralBrain")
            if brain_comp and brain_comp.get("is_possessed", 0.0) > 0.5:
                continue

            transform = registry.get_component_snapshot(eid, "Transform")
            energy = registry.get_component_snapshot(eid, "Energy")
            
            if not transform or not energy:
                continue

            # 1. Trích xuất véc-tơ đầu vào (Input Tensor)
            dx = target_x - transform["x"]
            dy = target_y - transform["y"]
            current_energy = energy["current_energy"]
            
            # Chuẩn hóa đầu vào để tránh bùng nổ gradient
            inputs = np.array([
                transform["x"] / 1000.0, 
                transform["y"] / 1000.0, 
                dx / 1000.0, 
                dy / 1000.0, 
                current_energy / 100.0,
                target_density / 100.0
            ], dtype=np.float32)

            # 2. Xử lý qua Mạng Nơ-ron (Forward Pass)
            brain = self.register_or_get_brain(eid)
            hidden = np.tanh(np.dot(inputs, brain.W1) + brain.b1)
            output = np.tanh(np.dot(hidden, brain.W2) + brain.b2) # Output [-1, 1]

            # 3. Ghi kết quả phản xạ cơ học vào Component Velocity
            v_max = 50.0 # Giới hạn tốc độ vĩ mô
            registry.set_component_attr(eid, "Velocity", "vx", float(output[0] * v_max))
            registry.set_component_attr(eid, "Velocity", "vy", float(output[1] * v_max))

    # =====================================================================
    # 2. QUY TRÌNH TIẾN HÓA (Genetic Algorithm / Generation End)
    # =====================================================================

    def evaluate_and_evolve(self, registry: Any) -> None:
        """
        Thực thi tiến hóa. Gọi khi một "Ngày/Thế hệ" trong mô phỏng kết thúc.
        """
        logger.info(f"[EvolutionEngine] Bắt đầu tính toán Tiến hóa Thế hệ {self.current_generation}")
        
        # 1. Đánh giá độ thích nghi (Fitness Evaluation)
        # NPC nào sống sót và tích lũy được nhiều Energy nhất sẽ được chọn làm cha mẹ
        fitness_scores: List[Tuple[int, float]] = []
        
        entities = registry.query("NeuralBrain", "Energy", "Health")
        for eid in entities:
            health = registry.get_component_snapshot(eid, "Health")
            energy = registry.get_component_snapshot(eid, "Energy")
            
            if health and health["is_alive"] > 0.5:
                # Fitness = Lượng năng lượng dư thừa
                score = energy["current_energy"]
                fitness_scores.append((eid, float(score)))
                registry.set_component_attr(eid, "NeuralBrain", "fitness_score", float(score))

        if len(fitness_scores) < 2:
            logger.warning("[EvolutionEngine] Quần thể diệt vong hoặc quá ít để lai tạo.")
            return

        # Sắp xếp từ cao xuống thấp
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Chọn lọc tự nhiên (Selection - Top 20% tinh hoa)
        elite_count = max(2, len(fitness_scores) // 5)
        elites = [eid for eid, score in fitness_scores[:elite_count]]
        
        logger.info(f"[EvolutionEngine] Chọn lọc {elite_count} tinh hoa. Fitness cao nhất: {fitness_scores[0][1]:.2f}")

        # 3. Lai ghép và Đột biến cho thế hệ mới (Crossover & Mutation)
        new_brains: Dict[int, BrainTopology] = {}
        
        for eid in entities:
            if eid in elites:
                # Tinh hoa được giữ nguyên cấu trúc bộ não (Bảo tồn gen tốt)
                new_brains[eid] = self.brains[eid].clone()
            else:
                # Những NPC yếu kém sẽ được "đầu thai" bằng cách lai ghép từ 2 tinh hoa ngẫu nhiên
                parent_a_id = random.choice(elites)
                parent_b_id = random.choice(elites)
                
                brain_a = self.brains[parent_a_id]
                brain_b = self.brains[parent_b_id]
                
                child_brain = self._crossover_and_mutate(brain_a, brain_b)
                new_brains[eid] = child_brain
                
                # Reset máu và năng lượng cho thế hệ mới
                registry.set_component_attr(eid, "Energy", "current_energy", 100.0)
                registry.set_component_attr(eid, "Health", "current_hp", 100.0)

        # Cập nhật quần thể
        self.brains = new_brains
        self.current_generation += 1
        
        # Cập nhật Generation lên ECS để Telemetry có thể theo dõi
        for eid in entities:
            registry.set_component_attr(eid, "NeuralBrain", "generation", float(self.current_generation))
            
        logger.info(f"[EvolutionEngine] Đã hoàn thành quá trình thay máu ma trận thần kinh. Chào mừng Thế hệ {self.current_generation}.")

    def _crossover_and_mutate(self, parent_a: BrainTopology, parent_b: BrainTopology) -> BrainTopology:
        """Lai ghép ma trận trọng số của hai bộ não và áp dụng đột biến xác suất."""
        child = BrainTopology(parent_a.input_dim, parent_a.hidden_dim, parent_a.output_dim)
        
        # 3.1 Lai ghép (Uniform Crossover - Trộn điểm ma trận)
        mask_W1 = np.random.rand(*parent_a.W1.shape) > 0.5
        child.W1 = np.where(mask_W1, parent_a.W1, parent_b.W1)
        
        mask_W2 = np.random.rand(*parent_a.W2.shape) > 0.5
        child.W2 = np.where(mask_W2, parent_a.W2, parent_b.W2)
        
        # Lai ghép Biases
        child.b1 = np.where(np.random.rand(*parent_a.b1.shape) > 0.5, parent_a.b1, parent_b.b1)
        child.b2 = np.where(np.random.rand(*parent_a.b2.shape) > 0.5, parent_a.b2, parent_b.b2)

        # 3.2 Đột biến (Gaussian Mutation)
        def mutate_matrix(mat: np.ndarray) -> np.ndarray:
            mutation_mask = np.random.rand(*mat.shape) < self.mutation_rate
            noise = np.random.randn(*mat.shape) * self.mutation_strength
            return mat + (mutation_mask * noise)

        child.W1 = mutate_matrix(child.W1)
        child.W2 = mutate_matrix(child.W2)
        child.b1 = mutate_matrix(child.b1)
        child.b2 = mutate_matrix(child.b2)

        return child
    
    