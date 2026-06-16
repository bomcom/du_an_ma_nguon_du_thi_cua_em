"""
evolution/soul_possession.py
============================
Agent Soul Possession Execution Layer (Cơ chế Nhập Hồn)

Architecture
------------
    Thực thi việc chuyển đổi luồng điều khiển (Control Thread Hijacking)
    từ Trí tuệ tự thân (Micro-Neural Network) sang Người chơi (User Input),
    nhưng vẫn duy trì áp lực sinh tồn phần cứng (LTspice).

    Workflow:
    1. POSSESS: Lưu trạng thái (Context Saving), ngắt kết nối mạng nơ-ron, 
       chuyển flag `is_possessed = 1.0`.
    2. RUNTIME (Tick): Đọc Input từ người dùng -> Tính toán v_max thực tế từ 
       LTspice Bandwidth -> Ánh xạ thành Vận tốc (Velocity) -> Ép tiêu hao Energy.
    3. RESTORE: Trả lại quyền điều khiển cho Mạng nơ-ron, reset cờ `is_possessed = 0.0`.
"""
"""

Tên tệp: soul_possession.py
mô tả: Cơ chế Nhập Hồn - Cho phép người dùng trực tiếp điều khiển một NPC trong hệ sinh thái, đồng thời vẫn duy trì áp lực sinh tồn từ phần cứng LTspice. 

Bản quyền © 2026 Phạm Hồng Hải Đăng.

Mọi quyền được bảo lưu.

Tài liệu này thuộc sở hữu trí tuệ của Phạm Hồng Hải Đăng.

"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("SoulPossession")

class SoulPossessionManager:
    """
    Quản lý trạng thái và luồng điều khiển khi Người dùng nhập vai NPC.
    """
    def __init__(self, ecs_registry: Any, telemetry_core: Any = None):
        """
        Parameters
        ----------
        ecs_registry   : Tham chiếu đến ECSRegistry để thao tác Component.
        telemetry_core : (Optional) Tham chiếu đến Telemetry để focus theo dõi NPC đang nhập hồn.
        """
        self.ecs = ecs_registry
        self.telemetry = telemetry_core
        self.possessed_entity_id: Optional[int] = None
        
        # Biến cấu hình để mapping Input
        # (Ví dụ: phím W/A/S/D sẽ tương đương giá trị [-1, 1] cho x, y)
        self.current_input_x: float = 0.0
        self.current_input_y: float = 0.0

    # =====================================================================
    # 1. API ĐIỀU KHIỂN LUỒNG (Hijack & Restore)
    # =====================================================================

    def possess_agent(self, entity_id: int) -> bool:
        """
        Kích hoạt cơ chế Nhập hồn vào một NPC cụ thể.
        """
        # Kiểm tra xem thực thể có tồn tại và có não bộ không
        if not self.ecs.entity_exists(entity_id):
            logger.error(f"[SoulPossession] Thực thể {entity_id} không tồn tại.")
            return False
            
        if not self.ecs.has_component(entity_id, "NeuralBrain"):
            logger.error(f"[SoulPossession] Thực thể {entity_id} không có NeuralBrain để chiếm quyền.")
            return False

        # Nếu đang nhập một NPC khác, phải thoát ra trước
        if self.possessed_entity_id is not None:
            self.restore_agent()

        # Đánh dấu cờ (Flag) trong Component để thuật toán Tiến hóa (GeneticEngine) bỏ qua NPC này
        self.ecs.set_component_attr(entity_id, "NeuralBrain", "is_possessed", 1.0)
        
        self.possessed_entity_id = entity_id
        
        # Nếu có Telemetry, tự động chuyển tiêu điểm theo dõi sang NPC này
        if self.telemetry:
            self.telemetry.set_focus_entity(entity_id)

        logger.info(f"[SoulPossession] >>> ĐÃ NHẬP HỒN THÀNH CÔNG VÀO THỰC THỂ {entity_id} <<<")
        logger.info("[SoulPossession] Mạng nơ-ron tự trị đã bị đóng băng. Đợi Input từ Người dùng.")
        return True

    def restore_agent(self) -> bool:
        """
        Thoát vai, trả lại quyền điều khiển cho mạng nơ-ron tiến hóa.
        """
        if self.possessed_entity_id is None:
            return False

        eid = self.possessed_entity_id
        
        # Trả lại cờ cho mạng nơ-ron
        if self.ecs.entity_exists(eid) and self.ecs.has_component(eid, "NeuralBrain"):
            self.ecs.set_component_attr(eid, "NeuralBrain", "is_possessed", 0.0)
            # Dừng NPC lại để nó tự quyết định bước tiếp theo
            self.ecs.set_component_attr(eid, "Velocity", "vx", 0.0)
            self.ecs.set_component_attr(eid, "Velocity", "vy", 0.0)

        logger.info(f"[SoulPossession] <<< ĐÃ THOÁT VAI KHỎI THỰC THỂ {eid} <<<")
        logger.info("[SoulPossession] Mạng nơ-ron đã được khôi phục quyền điều khiển.")
        
        self.possessed_entity_id = None
        self.current_input_x = 0.0
        self.current_input_y = 0.0
        
        if self.telemetry:
            self.telemetry.clear_focus()
            
        return True

    # =====================================================================
    # 2. GIAO DIỆN NHẬP DỮ LIỆU BÀN PHÍM/CHUỘT
    # =====================================================================

    def set_user_input(self, input_x: float, input_y: float) -> None:
        """
        Nhận tín hiệu từ Engine đồ họa (Pygame/Ursina).
        Giá trị input_x, input_y chuẩn hóa trong khoảng [-1.0, 1.0].
        Ví dụ: W -> y=1, S -> y=-1, A -> x=-1, D -> x=1.
        """
        if self.possessed_entity_id is None:
            return
            
        # Chuẩn hóa vectơ input để tránh đi chéo nhanh hơn đi thẳng
        length = np.sqrt(input_x**2 + input_y**2)
        if length > 1.0:
            input_x /= length
            input_y /= length
            
        self.current_input_x = float(input_x)
        self.current_input_y = float(input_y)

    # =====================================================================
    # 3. HÀM HỆ THỐNG GẮN VÀO ECS (System Executed Per Tick)
    # =====================================================================

    def manual_control_system(self, registry: Any, ltspice_state: Dict[str, Any], dt: float) -> None:
        """
        Hàm này được gọi mỗi Tick trong vòng lặp của SessionBox.
        Nó tính toán vận tốc dựa trên Input và áp lực từ phần cứng LTspice.
        """
        if self.possessed_entity_id is None:
            return

        eid = self.possessed_entity_id
        
        # Nếu NPC chết trong lúc đang nhập vai, cưỡng bức thoát
        health = registry.get_component_snapshot(eid, "Health")
        if health and health.get("is_alive", 0.0) < 0.5:
            logger.warning(f"[SoulPossession] Thực thể {eid} đã chết. Cưỡng bức thoát hồn.")
            self.restore_agent()
            return

        velocity = registry.get_component_snapshot(eid, "Velocity")
        energy = registry.get_component_snapshot(eid, "Energy")
        mass_comp = registry.get_component_snapshot(eid, "Mass")
        
        if not velocity or not energy:
            return

        # -------------------------------------------------------------
        # CƠ CHẾ RÀNG BUỘC 1: v_max ĐỊNH NGHĨA BỞI LTSPICE BANDWIDTH
        # -------------------------------------------------------------
        # Hệ số k_bw tương tự như trong matrix_solver (S_4)
        k_bw = 0.001 
        bw_hz = ltspice_state.get("bandwidth_hz", 1000.0)
        
        # Nếu thiết kế mạch kém (Băng thông thấp), NPC sẽ chạy rất chậm dù người chơi có ấn phím mạnh
        v_max = k_bw * bw_hz 
        
        # Ràng buộc thêm bởi khối lượng (NPC nặng thì chậm)
        mass = mass_comp.get("mass_kg", 1.0) if mass_comp else 1.0
        v_max_actual = max(1.0, v_max / mass)

        # -------------------------------------------------------------
        # ÁNH XẠ VẬN TỐC THỰC TẾ
        # -------------------------------------------------------------
        actual_vx = self.current_input_x * v_max_actual
        actual_vy = self.current_input_y * v_max_actual
        
        registry.set_component_attr(eid, "Velocity", "vx", float(actual_vx))
        registry.set_component_attr(eid, "Velocity", "vy", float(actual_vy))

        # -------------------------------------------------------------
        # CƠ CHẾ RÀNG BUỘC 2: TIÊU HAO NĂNG LƯỢNG KHI DI CHUYỂN
        # -------------------------------------------------------------
        # Người chơi di chuyển càng nhanh, lượng calo (Energy) tiêu hao càng lớn.
        # Nếu mạch bị nhiễu (SNR thấp), hao phí năng lượng (e_burn) càng cao.
        snr_db = ltspice_state.get("snr_db", 30.0)
        k_snr = 0.0001
        
        # Tính toán độ dời vật lý
        speed_sq = actual_vx**2 + actual_vy**2
        
        if speed_sq > 0.1:
            # Tiêu hao do vận động cơ học + nhiễu phần cứng
            movement_burn = (np.sqrt(speed_sq) * 0.05) * dt
            hardware_penalty = (mass * k_snr * snr_db) * dt
            total_burn = movement_burn + hardware_penalty
            
            current_e = energy.get("current_energy", 0.0)
            new_e = max(0.0, current_e - total_burn)
            registry.set_component_attr(eid, "Energy", "current_energy", float(new_e))

            