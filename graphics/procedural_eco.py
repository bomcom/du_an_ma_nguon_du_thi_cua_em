"""
graphics/procedural_eco.py
==========================
Procedural Generation Pipeline & Biomass Equilibrium Engine

Architecture
------------
    Sinh ra thế giới tự nhiên dựa trên hàm nhiễu liên tục (Perlin/Simplex Noise) 
    nhưng chịu sự chi phối tuyệt đối của Lõi Toán học (Matrix Solver - S1).
    
    Quy trình:
    1. Perlin Noise -> Biome Map (Nước, Cỏ, Rừng, Núi).
    2. Quét bản đồ để tìm các tọa độ có khả năng sinh trưởng (Forest biome).
    3. Biomass Ceiling: Tính toán tổng năng lượng dự kiến. Nếu vượt quá E_max của S1,
       áp dụng thuật toán cắt tỉa (Culling) ngẫu nhiên để ép hệ sinh thái về trạng thái cân bằng.
    4. Decay Algorithm: Thực vật/Năng lượng không được tiêu thụ sẽ bị phân rã theo thời gian (Entropy).
"""

import logging
import random
import math
from typing import Any, Dict, List, Tuple

import numpy as np
# Lưu ý: Trong môi trường thực tế, bạn có thể `pip install noise` để dùng snoise2/pnoise2.
# Ở đây ta giả lập một hàm nhiễu cơ bản bằng numpy để module có thể chạy độc lập.

logger = logging.getLogger("ProceduralEcosystem")

class ProceduralEcosystem:
    """
    Hệ thống sinh thái thủ tục bị giới hạn bởi Toán học vĩ mô.
    """
    def __init__(self, ecs_registry: Any, width: int = 1000, height: int = 1000, seed: int = 42):
        self.ecs = ecs_registry
        self.width = width
        self.height = height
        self.seed = seed
        self.grid_size = 20 # Kích thước mỗi ô (Grid cell)
        
        # Thiết lập seed tĩnh để đảm bảo tính Deterministic (Xác định)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.biome_map: np.ndarray = np.zeros((self.width // self.grid_size, self.height // self.grid_size))
        self.active_flora_entities: List[int] = []

    # =====================================================================
    # 1. PERLIN NOISE & BIOME MAPPING
    # =====================================================================

    def _generate_pseudo_noise(self, x: float, y: float, scale: float = 0.1) -> float:
        """Hàm giả lập nhiễu 2D (Thay thế cho Perlin/Simplex nếu không có thư viện)."""
        # Sử dụng giao thoa sóng Sine/Cosine để tạo địa hình liên tục
        val = math.sin(x * scale + self.seed) * math.cos(y * scale + self.seed)
        val += 0.5 * math.sin(x * scale * 2.0) * math.cos(y * scale * 2.0)
        return val # Trả về giá trị trong khoảng xấp xỉ [-1.5, 1.5]

    def generate_biome_map(self) -> None:
        """Tạo bản đồ phân bố quần xã sinh vật (Biome)."""
        logger.info(f"[EcoSystem] Đang khởi sinh địa hình với Seed: {self.seed}...")
        cols, rows = self.biome_map.shape
        
        for i in range(cols):
            for j in range(rows):
                # Scale tọa độ để nhiễu mịn hơn
                nx = i / cols * 10.0
                ny = j / rows * 10.0
                noise_val = self._generate_pseudo_noise(nx, ny)
                self.biome_map[i, j] = noise_val

    # =====================================================================
    # 2. BIOMASS CEILING (GIỚI HẠN TỪ S1 CỦA LÕI TOÁN)
    # =====================================================================

    def spawn_ecosystem(self, matrix_solver_state: Dict[str, Any]) -> None:
        """
        Khởi tạo thực vật dựa trên Biome Map, nhưng bị giới hạn bởi E_max.
        """
        self.generate_biome_map()
        
        # 1. Đọc giới hạn vĩ mô từ hệ kín S1
        # Nếu mạch có Bandwidth/SNR thấp, E_max sẽ bị thu hẹp, thế giới trở nên cằn cỗi
        e_max = matrix_solver_state.get("E_max", 5000.0) 
        energy_per_tree = 50.0
        max_allowed_trees = int(e_max / energy_per_tree)
        
        logger.info(f"[EcoSystem] Giới hạn sinh khối (S1) cho phép: {e_max} Calories (~{max_allowed_trees} cây).")

        # 2. Quét bản đồ để tìm tọa độ tiềm năng (Ví dụ: Noise > 0.3 là Rừng)
        potential_sites: List[Tuple[int, int]] = []
        cols, rows = self.biome_map.shape
        
        for i in range(cols):
            for j in range(rows):
                if self.biome_map[i, j] > 0.3: # Ngưỡng sinh thái Rừng
                    real_x = i * self.grid_size
                    real_y = j * self.grid_size
                    potential_sites.append((real_x, real_y))

        # 3. Thuật toán Cắt tỉa (Culling) dựa trên Trần sinh khối
        total_potential = len(potential_sites)
        if total_potential > max_allowed_trees:
            survival_rate = max_allowed_trees / total_potential
            logger.warning(f"[EcoSystem] Nhu cầu ({total_potential}) vượt Giới hạn ({max_allowed_trees}). Ép tỉ lệ sinh tồn: {survival_rate*100:.1f}%")
            # Lọc ngẫu nhiên giữ lại đúng số lượng cây mà năng lượng S1 cho phép
            approved_sites = [site for site in potential_sites if random.random() < survival_rate]
        else:
            approved_sites = potential_sites

        # 4. Đăng ký Thực thể (Flora) vào ECS
        for x, y in approved_sites:
            flora_id = self.ecs.create_entity()
            
            # Thêm độ lệch ngẫu nhiên nhỏ để cây không xếp hàng thẳng tắp
            offset_x = x + random.uniform(-5, 5)
            offset_y = y + random.uniform(-5, 5)
            
            self.ecs.add_component(flora_id, "Transform", {"x": offset_x, "y": offset_y, "rotation": 0.0})
            self.ecs.add_component(flora_id, "Energy", {"current_energy": energy_per_tree, "max_energy": energy_per_tree})
            self.ecs.add_component(flora_id, "FloraDef", {"type": "tree", "decay_rate": 0.1})
            
            self.active_flora_entities.append(flora_id)
            
        logger.info(f"[EcoSystem] Khởi sinh hoàn tất. Số lượng thực vật thực tế: {len(self.active_flora_entities)}")

    # =====================================================================
    # 3. DECAY ALGORITHM (ĐỊNH LUẬT ENTROPY - VẬN HÀNH THEO TICK)
    # =====================================================================

    def decay_system(self, dt: float) -> None:
        """
        Hàm System chạy mỗi khung hình. 
        Mô phỏng sự hao hụt năng lượng tự nhiên (Entropy).
        Thực vật không được ăn sẽ dần thối rữa và biến mất.
        """
        dead_flora = []
        
        for eid in self.active_flora_entities:
            energy_comp = self.ecs.get_component_snapshot(eid, "Energy")
            flora_comp = self.ecs.get_component_snapshot(eid, "FloraDef")
            
            if not energy_comp or not flora_comp:
                continue
                
            # Phân rã năng lượng dựa trên thời gian (dt) và hệ số rã (decay_rate)
            decay_amount = flora_comp.get("decay_rate", 0.1) * dt
            new_energy = energy_comp["current_energy"] - decay_amount
            
            if new_energy <= 0:
                dead_flora.append(eid)
            else:
                self.ecs.set_component_attr(eid, "Energy", "current_energy", float(new_energy))

        # Dọn dẹp xác thực vật đã cạn kiệt năng lượng khỏi ECS
        for dead_id in dead_flora:
            self.ecs.destroy_entity(dead_id)
            self.active_flora_entities.remove(dead_id)
            
        if dead_flora:
            logger.debug(f"[EcoSystem] {len(dead_flora)} thực thể sinh học đã phân rã hoàn toàn do Entropy.")

            