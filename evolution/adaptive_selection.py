# -*- coding: utf-8 -*-
"""
evolution/adaptive_selection.py
==============================
Hệ Thống Chọn Lọc Tiến Hóa Thích Nghi Động (Dynamic Adaptive Selection Engine)
Kiến trúc: Trait-Agnostic Engine (Độc lập cấu trúc tính trạng)

Mô tả:
    File này hiện thực hóa Giai đoạn 4A của Đề án. Nó hoạt động như một bộ lọc tự nhiên
    vĩ mô, tự động bóc tách chuỗi nhiễm sắc thể từ ECSRegistry thông qua cơ chế phản chiếu,
    chấm điểm Fitness dựa trên áp lực toán học từ MatrixSolver và thực hiện nhân bản, 
    phối giống cấu trúc, đột biến liên tục mà không giả định trước bất kỳ Component nào cố định.

Tác giả: Chuyên gia Kiến trúc Hệ thống Mô phỏng
"""

from __future__ import annotations
import random
import copy
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger("SimulationKernel.Evolution")


# ==========================================================
# DATA STRUCTURES & SNAPSHOTS
# ==========================================================

@dataclass
class AdaptiveGenomeSnapshot:
    """
    Đóng gói dữ liệu di truyền thô và chỉ số sinh tồn của một thực thể tại một nhịp kiểm thử.
    genome lưu trữ một deepcopy cấu trúc dict component lấy từ profile của ECSRegistry.
    """
    entity_id: int
    fitness: float
    genome: Dict[str, Dict[str, float]]
    generation: int


@dataclass
class FitnessConfig:
    """Hệ tham số cấu hình trọng số sinh học và hình phạt toán học toàn cục"""
    weight_energy: float = 1.2
    weight_health: float = 1.5
    weight_age: float = 0.4
    
    # Hình phạt nghiêm khắc nếu thực thể vi phạm các định luật bảo toàn hệ kín của MatrixSolver
    penalty_violation: float = 100.0
    
    # Tham số điều phối thuật toán di truyền tiến hóa
    elite_ratio: float = 0.15          # Tỷ lệ giữ lại nhóm tinh hoa cấp cao
    tournament_size: float = 4         # Quy mô giải đấu chọn lọc tự nhiên
    mutation_rate: float = 0.08        # Xác suất xảy ra biến dị trên mỗi thuộc tính số
    mutation_strength: float = 0.12    # Biên độ nhiễu Gauss áp vào biến dị


@dataclass
class EvolutionStats:
    """Bộ lưu trữ chỉ số đo lường Telemetry phục vụ giám sát tiến trình tiến hóa"""
    current_generation: int = 0
    max_fitness: float = 0.0
    mean_fitness: float = 0.0
    population_size: int = 0
    diversity_index: float = 0.0
    violation_penalty_count: int = 0


# ==========================================================
# MUTATION CONSTRAINT LAYER (MÀNG LỌC BIẾN DỊ AN TOÀN)
# ==========================================================

class MutationConstraintLayer:
    """
    Chốt chặn kiểm soát biên độ biến dị. Đảm bảo các đột biến số học ngẫu nhiên
    không đẩy các thông số vật lý/sinh học về trạng thái phi vật lý (âm hoặc tràn ngưỡng).
    """
    
    @staticmethod
    def clamp_attribute(component_name: str, attr_name: str, value: float) -> float:
        """Áp đặt giới hạn cứng (Boundary Conditions) cho từng thuộc tính đặc thù"""
        # Quy luật bảo toàn khối lượng và hình học cơ bản
        if attr_name in ("mass_kg", "inv_mass", "current_health", "max_health", "current_energy", "max_capacity"):
            return max(1e-4, value)
            
        # Giới hạn tỷ lệ hấp thụ/tiêu hao (như của HVE, Mana,...)
        if attr_name in ("absorb_rate", "consumption_rate", "mutation_rate"):
            return max(0.0, min(1.0, value))
            
        # Mặc định giữ nguyên nếu là các tọa độ vật lý hoặc vận tốc tự do
        return value


# ==========================================================
# ADAPTIVE SELECTION ENGINE (LÕI ĐIỀU PHỐI TIẾN HÓA THÍCH NGHI)
# ==========================================================

class AdaptiveSelectionEngine:
    """
    Nhà nhạc trưởng điều phối toàn bộ chu kỳ sinh sản, lai ghép cấu trúc mở rộng,
    và chọn lọc tự nhiên của thực thể dưới áp lực môi trường toán học.
    """

    def __init__(
        self,
        registry: Any,
        matrix_solver: Any,
        config: Optional[FitnessConfig] = None
    ):
        """
        Khởi tạo Engine tiến hóa động tương thích sâu với ECSRegistry và MatrixSolver.
        """
        self.registry = registry
        self.solver = matrix_solver
        self.cfg = config if config is not None else FitnessConfig()
        
        self.generation_counter = 0
        self.history_stats: List[EvolutionStats] = []
        self._lock = threading.RLock()  # Bảo vệ an toàn luồng tuyệt đối khi chạy đa luồng

        logger.info("[EvolutionEngine] Kích hoạt thành công phân hệ Tiến hóa Thích nghi Động Giai đoạn 4A.")

    # ------------------------------------------------------
    # FITNESS EVALUATION PIPELINE
    # ------------------------------------------------------

    def evaluate_entity_fitness(self, entity_id: int) -> float:
        """
        TẦNG 1, 2, 3: Chấm điểm độ thích nghi thực tế dựa trên dữ liệu trích xuất từ ECS.
        Tuyệt đối không giả định trước sự tồn tại của các Component động (HVE, Mana, SoulLink).
        """
        fitness = 0.0

        # 1. Trích xuất thành phần Năng lượng (Energy Component) bằng API Snapshot an toàn
        energy_snap = self.registry.get_component_snapshot(entity_id, "Energy")
        if energy_snap:
            # Ưu tiên lấy trường năng lượng chuẩn từ hệ thống của bạn
            current_e = float(energy_snap.get("current_energy", energy_snap.get("energy", 0.0)))
            fitness += current_e * self.cfg.weight_energy

        # 2. Trích xuất thành phần Sức khỏe Sinh học (Health Component)
        health_snap = self.registry.get_component_snapshot(entity_id, "Health")
        if health_snap:
            current_h = float(health_snap.get("current_health", health_snap.get("hp", 0.0)))
            fitness += current_h * self.cfg.weight_health

        # 3. Trích xuất Tuổi thọ (Age Component nếu hệ thống sinh thế giới có tích hợp)
        age_snap = self.registry.get_component_snapshot(entity_id, "Age")
        if age_snap:
            current_age = float(age_snap.get("age", 0.0))
            fitness += current_age * self.cfg.weight_age

        # 4. KIỂM TRA HỆ KÍN TOÁN HỌC: Trừng phạt nếu rách lưới bảo toàn của MatrixSolver
        try:
            violations = self.solver.get_violations()
            if violations and any(violations.values()):
                fitness -= self.cfg.penalty_violation
                # Đánh dấu vết để ghi log telemetry vĩ mô
                if hasattr(self, '_current_tick_violation_count'):
                    self._current_tick_violation_count += 1
        except Exception as e:
            logger.debug(f"[Evolution] Bỏ qua kiểm tra violation do cấu hình Solver: {e}")

        # Độ thích nghi sinh học không bao giờ được âm
        return max(0.0, fitness)

    # ------------------------------------------------------
    # GENOME REFLECTION INTERFACE
    # ------------------------------------------------------

    def build_genome_snapshot(self, entity_id: int) -> Optional[AdaptiveGenomeSnapshot]:
        """
        TẦNG THẨM ĐỊNH DI TRUYỀN: Duyệt quét động toàn bộ hồ sơ linh hồn của Entity.
        Nếu người dùng nạp một Component mới ở Runtime, nó lập tức biến thành một chuỗi gen di truyền.
        """
        with self._lock:
            # Sử dụng API chuẩn get_entity_profile của bạn để lấy toàn bộ danh sách component của thực thể
            profile = self.registry.get_entity_profile(entity_id)
            if not profile:
                return None

            genome_data = {}
            for comp_type, comp_data in profile.items():
                # Loại bỏ các định danh quản lý bộ nhớ nội bộ của ECS Registry
                if comp_type in ("entity_id", "id", "type_name", "scope", "mutability"):
                    continue
                
                # Thực hiện nhân bản sâu (Deepcopy) để cô lập dữ liệu ô nhớ tĩnh trong ECS Registry
                if isinstance(comp_data, dict):
                    genome_data[comp_type] = copy.deepcopy(comp_data)

            fitness_score = self.evaluate_entity_fitness(entity_id)

            return AdaptiveGenomeSnapshot(
                entity_id=entity_id,
                fitness=fitness_score,
                genome=genome_data,
                generation=self.generation_counter
            )

    # ------------------------------------------------------
    # DARWINIAN SELECTION MECHANISMS
    # ------------------------------------------------------

    def _select_tournament_parent(self, population: List[AdaptiveGenomeSnapshot]) -> AdaptiveGenomeSnapshot:
        """Lọc tuyển sinh học qua giải đấu nhỏ (Tournament Selection) để tránh bẫy tối ưu cục bộ"""
        candidates = random.sample(population, min(len(population), int(self.cfg.tournament_size)))
        # Kẻ có điểm fitness cao nhất sẽ thắng giải đấu và được quyền phối giống
        return max(candidates, key=lambda ind: ind.fitness)

    def _extract_elites(self, population: List[AdaptiveGenomeSnapshot]) -> List[AdaptiveGenomeSnapshot]:
        """Trích xuất nhóm tinh hoa (Elite Selection) bảo toàn nguyên vẹn mã gen trội sang thế hệ sau"""
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        elite_count = max(2, int(len(population) * self.cfg.elite_ratio))
        return population[:elite_count]

    # ------------------------------------------------------
    # STRUCTURAL SET-UNION CROSSOVER
    # ------------------------------------------------------

    def crossover(self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        TẦNG LAI GHÉP ĐỘNG: Tạo ra sự kết hợp kiến trúc nhiễm sắc thể.
        Giải quyết triệt để bài toán: Nếu bố không có HVE nhưng mẹ có HVE (do Prompt inject), con sẽ có cơ hội nhận HVE.
        """
        child_genome = {}
        
        # Lấy phép HỢP (Set Union) của tất cả các loại Component có trong cả cha lẫn mẹ
        all_component_types = set(parent_a.keys()).union(set(parent_b.keys()))

        for comp_type in all_component_types:
            # Trường hợp 1: Cả hai cùng có chung Component -> Trộn ngẫu nhiên hoặc lấy nguyên khối từ một bên
            if comp_type in parent_a and comp_type in parent_b:
                if random.random() > 0.5:
                    child_genome[comp_type] = copy.deepcopy(parent_a[comp_type])
                else:
                    child_genome[comp_type] = copy.deepcopy(parent_b[comp_type])
            
            # Trường hợp 2: Chỉ một bên có (Ví dụ: Tính trạng HVE được người dùng kích hoạt riêng cho một cụm sinh vật)
            elif comp_type in parent_a:
                child_genome[comp_type] = copy.deepcopy(parent_a[comp_type])
            else:
                child_genome[comp_type] = copy.deepcopy(parent_b[comp_type])

        return child_genome

    # ------------------------------------------------------
    # CONTINUOUS CONSTRAINT MUTATION
    # ------------------------------------------------------

    def mutate_genome(self, genome: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        TẦNG BIẾN DỊ SỐ HỌC: Dò quét toàn bộ ma trận số thực của mọi Component để gây đột biến.
        Kiểm soát chặt chẽ qua màng lọc MutationConstraintLayer để tránh lỗi sụp đổ toán học.
        """
        mutated_genome = copy.deepcopy(genome)

        for comp_name, attributes in mutated_genome.items():
            if not isinstance(attributes, dict):
                continue
                
            for attr_key, attr_val in attributes.items():
                # Chỉ gây biến dị lên các biến định lượng toán học (float, int)
                if not isinstance(attr_val, (int, float)):
                    continue
                
                # Áp tỷ lệ kích hoạt đột biến ngẫu nhiên
                if random.random() > self.cfg.mutation_rate:
                    continue

                # Tạo nhiễu biến dị Gauss (Normal Distribution)
                noise = np.random.normal(0.0, self.cfg.mutation_strength)
                new_val = attr_val + noise

                # Ép giá trị sau biến dị đi qua màng lọc ràng buộc cứng để bảo vệ hệ thống
                attributes[attr_key] = MutationConstraintLayer.clamp_attribute(comp_name, attr_key, new_val)

        return mutated_genome

    # ------------------------------------------------------
    # GENERATION CYCLE RUNTIME (CORE WORKFLOW)
    # ------------------------------------------------------

    def reproduce_generation(self) -> List[Dict[str, Dict[str, float]]]:
        """
        TRÁI TIM CỦA ENGINE: Thực thi trọn vẹn vòng đời đóng gói di truyền.
        Trả ra danh sách Hàng đợi Gen Con cái (Offspring Genome Queue) chuẩn xác.
        Bàn giao dữ liệu sạch cho Phân hệ 4B (dynamic_genome.py) tiêu thụ.
        """
        with self._lock:
            self._current_tick_violation_count = 0
            
            # 1. Gọi API của bạn để lấy toàn bộ ID thực thể đang sống sót trên bản đồ ECS
            active_entity_ids = self.registry.get_all_entities()
            population_size = len(active_entity_ids)

            if population_size < 4:
                logger.warning(f"[Evolution] Quần thể quá ít ({population_size} cá thể). Không đủ điều kiện kích hoạt chu kỳ tiến hóa.")
                # Trả về mã gen của những kẻ đang sống để bảo toàn chủng tộc nền
                return [
                    snap.genome
                    for eid in active_entity_ids
                    for snap in [self.build_genome_snapshot(eid)]
                    if snap is not None
                ]

            # 2. Xây dựng mảng nhiễm sắc thể Snapshot động và tính toán điểm số thích nghi
            population_snapshots: List[AdaptiveGenomeSnapshot] = []
            for eid in active_entity_ids:
                snap = self.build_genome_snapshot(eid)
                if snap:
                    population_snapshots.append(snap)

            # 3. Trích xuất nhóm tinh hoa di truyền nền tảng
            elites = self._extract_elites(population_snapshots)
            
            # Thu thập dữ liệu phục vụ vẽ đồ thị Telemetry thời gian thực
            fitness_array = np.array([ind.fitness for ind in population_snapshots])
            stats = EvolutionStats(
                current_generation=self.generation_counter,
                max_fitness=float(np.max(fitness_array)),
                mean_fitness=float(np.mean(fitness_array)),
                population_size=population_size,
                diversity_index=self._calculate_genetic_diversity(population_snapshots),
                violation_penalty_count=self._current_tick_violation_count
            )
            self.history_stats.append(stats)
            
            logger.info(
                f"[Generation #{self.generation_counter}] "
                f"Max Fitness: {stats.max_fitness:.2f} | "
                f"Mean Fitness: {stats.mean_fitness:.2f} | "
                f"Chủng tộc đa dạng: {stats.diversity_index:.4f}"
            )

            # 4. Kích hoạt luồng sản sinh con cái bù đắp quy mô hệ thống
            offspring_genome_queue: List[Dict[str, Dict[str, float]]] = []
            
            # Bảo lưu trực tiếp cấu trúc gen của nhóm tinh hoa sang thế hệ sau
            for elite in elites:
                offspring_genome_queue.append(copy.deepcopy(elite.genome))

            # Phối giống lai ghép cho đến khi lấp đầy quy mô thế giới cũ
            while len(offspring_genome_queue) < population_size:
                # Tuyển lọc cha mẹ thông qua Giải đấu sinh tồn (Tournament Selection)
                parent_a = self._select_tournament_parent(population_snapshots)
                parent_b = self._select_tournament_parent(population_snapshots)
                
                # Ngăn chặn phối giống đồng huyết tuyệt đối nếu quy mô quần thể cho phép
                if parent_a.entity_id == parent_b.entity_id and len(elites) > 2:
                    parent_b = random.choice([e for e in elites if e.entity_id != parent_a.entity_id])

                # Tiến hành lai ghép hợp cấu trúc dữ liệu mở rộng
                child_genome = self.crossover(parent_a.genome, parent_b.genome)
                
                # Tiến hành đột biến mù trên toàn diện trường dữ liệu số thực
                child_genome = self.mutate_genome(child_genome)
                
                offspring_genome_queue.append(child_genome)

            # Tăng biến đếm thế hệ vĩ mô của hệ kín
            self.generation_counter += 1
            
            # Trả về hàng đợi gen sạch cho bộ Spawn thực thể xử lý ở bước sau
            return offspring_genome_queue

    # ------------------------------------------------------
    # TELEMETRY ANALYSIS HELPER
    # ------------------------------------------------------

    def _calculate_genetic_diversity(self, population: List[AdaptiveGenomeSnapshot]) -> float:
        """Đo lường độ đa dạng sinh học thông qua khoảng cách phân tách cấu trúc tổ hợp Component"""
        if not population:
            return 0.0
        
        # Thống kê tập hợp các loại component xuất hiện trong quần thể
        structural_hashes = []
        for ind in population:
            # Tạo một chuỗi định danh đại diện cho cấu trúc gen của cá thể đó
            sorted_components = sorted(list(ind.genome.keys()))
            structural_hashes.append("-".join(sorted_components))
            
        unique_structures = set(structural_hashes)
        # Tỷ lệ chủng tộc độc bản xuất hiện trên tổng quy mô sinh vật
        return len(unique_structures) / len(population)

    def get_latest_stats(self) -> Dict[str, Any]:
        """Xuất dữ liệu Snapshot phục vụ hiển thị trực quan hóa trên TelemetryCore Dashboard"""
        if not self.history_stats:
            return {"generation": self.generation_counter, "max_fitness": 0.0, "mean_fitness": 0.0}
        last = self.history_stats[-1]
        return {
            "generation": last.current_generation,
            "max_fitness": last.max_fitness,
            "mean_fitness": last.mean_fitness,
            "population_size": last.population_size,
            "diversity_index": last.diversity_index,
            "violation_rate": last.violation_penalty_count
        }
    