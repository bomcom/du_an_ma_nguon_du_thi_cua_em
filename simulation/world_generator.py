# simulation/world_generator.py
import random
import logging
import re
from typing import Dict, Any, Tuple

logger = logging.getLogger("WorldGenerator")

class WorldGenerator:
    """
    HỆ KHỞI SINH THEO QUY TRÌNH (Procedural Generation Pipeline)
    Nhiệm vụ: Dịch dịch cấu trúc JSON World Schema thành các thực thể vật lý trong ECS Registry.
    Kiểm soát nghiêm ngặt bởi Thread Lock và Giới hạn Ma trận Năng lượng.
    """
    def __init__(self, registry: Any, matrix_solver: Any = None) -> None:
        self.registry = registry
        self.solver = matrix_solver # Kết nối lõi toán để kiểm tra áp lực hệ sinh thái
        self.world_width = 900
        self.world_height = 900

    def generate_from_schema(self, schema: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            entities_groups = schema.get("entities", [])
            total_spawned = 0
            
            # 1. TIỀN THẨM ĐỊNH: Tính toán sơ bộ áp lực năng lượng trước khi sinh
            estimated_energy = 0.0
            for group in entities_groups:
                count = int(group.get("count", 1))
                energy_data = group.get("components", {}).get("Energy", {})
                estimated_energy += count * float(energy_data.get("current_energy", 100.0))

            if self.solver and (estimated_energy > self.solver.E_max):
                return False, f"Khởi sinh thất bại: Tổng năng lượng dự kiến ({estimated_energy}) vượt ngưỡng E_max của hệ kín!"

            # 2. KHÓA CẤU TRÚC & KHỞI SINH (Thread-Safe Injection)
            with self.registry._structural_lock:
                for group in entities_groups:
                    count = int(group.get("count", 1))
                    component_defs = group.get("components", {})

                    for _ in range(count):
                        eid = self.registry.create_entity()

                        for comp_name, comp_data in component_defs.items():
                            # Trích xuất giá trị ghi đè hoặc giữ nguyên mặc định
                            overrides = dict(comp_data) if comp_data else {}

                            # Tự động rải tọa độ ngẫu nhiên không gian nếu là Transform
                            if comp_name == "Transform":
                                overrides.setdefault("x", random.uniform(20, self.world_width - 20))
                                overrides.setdefault("y", random.uniform(20, self.world_height - 20))
                            if comp_name == "Velocity":
                                overrides.setdefault("vx", random.uniform(-2.0, 2.0))
                                overrides.setdefault("vy", random.uniform(-2.0, 2.0))

                            # Thêm component vào thực thể thực tế
                            self.registry.add_component(eid, comp_name, overrides)
                        
                        total_spawned += 1

            logger.info(f"[WorldGenerator] Đồng hóa thành công {total_spawned} thực thể vào không gian thực tại.")
            return True, f"Thành công: Đã sinh {total_spawned} thực thể động."

        except Exception as e:
            logger.error(f"[WorldGenerator] Sụp đổ quy trình sinh thế giới: {e}")
            return False, f"Lỗi Runtime: {str(e)}"

    def clear_world(self) -> None:
        """
        Tẩy sạch thế giới phục vụ cơ chế Reset thế giới đóng an toàn.
        """
        with self.registry._structural_lock:
            entities = list(self.registry.get_all_entities())
            for eid in entities:
                self.registry.destroy_entity(eid)
        logger.info("[WorldGenerator] Trạng thái không gian đã được làm sạch hoàn toàn (World Cleared).")

    def _default_entity_components(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Transform": {},
            "Velocity": {},
            "Mass": {"mass_kg": 2.0},
            "Health": {},
            "Energy": {},
            "NeuralBrain": {},
        }

    def _schema_from_prompt(self, prompt: str, default_count: int) -> Dict[str, Any]:
        match = re.search(r"\b(\d+)\b", prompt)
        count = int(match.group(1)) if match else default_count
        return {
            "world_name": "Prompt Genesis",
            "entities": [
                {
                    "count": max(1, min(count, 500)),
                    "components": self._default_entity_components(),
                }
            ],
        }

    async def generate_from_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Spawn a world population from a natural-language prompt."""
        schema = self._schema_from_prompt(prompt, default_count=50)
        return self.generate_from_schema(schema)

    async def spawn_entities_from_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Spawn additional entities from a natural-language prompt."""
        schema = self._schema_from_prompt(prompt, default_count=10)
        return self.generate_from_schema(schema)