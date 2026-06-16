"""
Filename: ai_core/ml_perception_engine.py
Description: ML_PERCEPTION_ENGINE — Phân hệ nhận thức hạ tầng vĩ mô.
             Trích xuất, tổng hợp dữ liệu cấu trúc động từ ECS Registry và Math State 
             để biên dịch thành Vector đặc trưng 12 chiều phục vụ kiểm định logic.
Author: Chuyên gia phần mềm AI
"""

import logging
import numpy as np


logger = logging.getLogger("MLPerceptionEngine")

class MLPerceptionEngine:
    """
    Bộ dịch nhận thức hệ thống (System Perception Engine).
    Chuyển đổi trạng thái phân tán của ECS (Môi trường vi mô) và Lõi toán (Môi trường vĩ mô)
    thành dạng biểu diễn vector toán học tuyến tính hóa (Linearized Feature Vector).
    """
    
    def __init__(self):
        logger.info("Khởi tạo ML_PERCEPTION_ENGINE thành công. Sẵn sàng bóc tách vector đặc trưng.")

    def extract_feature_vector(self, registry, math_state: dict, hardware_state: dict) -> np.ndarray:
        """
        Quét toàn bộ ECS Registry và Math State để sinh vector đặc trưng 12-dim float32.
        
        Mô hình ánh xạ 12 chiều (12-Dimensional Mapping):
        [0]: E_ratio             - Tỷ lệ năng lượng toàn hệ thống (E_total / E_max)
        [1]: M_ratio             - Tỷ lệ sinh khối/khối lượng (M_total / M_max)
        [2]: prey_norm           - Mật độ con mồi được chuẩn hóa (Prey Count / Base)
        [3]: predator_norm       - Mật độ kẻ săn mồi được chuẩn hóa (Predator Count / Base)
        [4]: v_max_norm          - Vận tốc tối đa của thực thể sở hữu (Ánh xạ từ Bandwidth phần cứng)
        [5]: e_burn_norm         - Tỷ lệ tiêu hao năng lượng hệ thống được chuẩn hóa
        [6]: lorenz_x_n          - Tọa độ X chuẩn hóa của phương trình Lorenz Attractor
        [7]: lorenz_y_n          - Tọa độ Y chuẩn hóa của phương trình Lorenz Attractor
        [8]: lorenz_z_n          - Tọa độ Z chuẩn hóa của phương trình Lorenz Attractor
        [9]: decay_flag          - Cờ kích hoạt suy thoái hệ thống (0.0 hoặc 1.0)
        [10]: pop_collapse_flag  - Cờ sụp đổ quần thể cục bộ (0.0 hoặc 1.0)
        [11]: raw_violation_cnt  - Tổng số lượng vi phạm ràng buộc thô trong chu kỳ trước
        """
        
        # --- BƯỚC 1: TRÍCH XUẤT VÀ TỔNG HỢP VI MÔ TỪ ECS REGISTRY ---
        total_energy_ecs = 0.0
        total_mass_ecs = 0.0
        entity_count = 0
        max_velocity_encountered = 0.0
        
        # Khóa an toàn luồng khi truy vấn dữ liệu từ registry dùng chung
        with registry.lock:
            active_entities = list(registry.entities)
            
            for eid in active_entities:
                entity_count += 1
                
                # Bóc tách cấu phần Năng lượng (Energy Component)
                energy_comp = registry.get_component_snapshot(eid, "Energy")
                total_energy_ecs += float(energy_comp.get("current_energy", 0.0))
                
                # Bóc tách cấu phần Sinh khối/Khối lượng (Mass Component)
                mass_comp = registry.get_component_snapshot(eid, "Mass")
                total_mass_ecs += float(mass_comp.get("mass_kg", 0.0))
                
                # Bóc tách cấu phần Động học (Velocity & Transform)
                vel_comp = registry.get_component_snapshot(eid, "Velocity")
                vx = float(vel_comp.get("vx", 0.0))
                vy = float(vel_comp.get("vy", 0.0))
                
                # Tính vận tốc cơ học thực tế theo định lý Pitago v = sqrt(vx^2 + vy^2)
                v_magnitude = np.sqrt(vx**2 + vy**2)
                if v_magnitude > max_velocity_encountered:
                    max_velocity_encountered = v_magnitude

        # --- BƯỚC 2: TÍNH TOÁN CÁC CHỈ SỐ CHUẨN HÓA (NORMALIZATION TIER) ---
        # 1. Tính toán Tỷ lệ Năng lượng (E_ratio)
        e_max = float(math_state.get("E_max", 1200.0))
        # Sử dụng năng lượng tính toán thực tế từ ECS, nếu trống sẽ lấy từ math_state vĩ mô
        actual_e_total = total_energy_ecs if total_energy_ecs > 0 else float(math_state.get("E_total", 0.0))
        e_ratio = actual_e_total / e_max if e_max > 0 else 0.0
        
        # 2. Tính toán Tỷ lệ Sinh khối (M_ratio)
        m_max = float(math_state.get("M_max", 5000.0))
        actual_m_total = total_mass_ecs if total_mass_ecs > 0 else float(math_state.get("M_total", 0.0))
        m_ratio = actual_m_total / m_max if m_max > 0 else 0.0
        
        # 3. Chuẩn hóa mật độ sinh thái (Prey / Predator normalization)
        prey_norm = float(math_state.get("prey", 0.0)) / 100.0
        predator_norm = float(math_state.get("predator", 0.0)) / 50.0
        
        # 4. Chuẩn hóa giới hạn vận tốc dựa trên DNA phần cứng (Bandwidth & SNR)
        # Sử dụng thông số phần cứng từ hardware_state để kiểm soát trần vận tốc lý thuyết
        hardware_bw = float(hardware_state.get("bandwidth", 25000.0))
        v_max_theoretical = hardware_bw / 1000.0
        v_max_norm = max_velocity_encountered / v_max_theoretical if v_max_theoretical > 0 else 0.0
        
        # 5. Chuẩn hóa hệ số đốt cháy năng lượng
        e_burn_norm = float(math_state.get("e_burn_rate", 1.0)) / 10.0
        
        # 6. Chuẩn hóa hệ động lực học Lorenz Attractor phi tuyến (Ép dải về khoảng [-1, 1] hoặc gần tương đương)
        lorenz_tuple = math_state.get("lorenz", (0.0, 0.0, 0.0))
        lorenz_x_n = float(lorenz_tuple[0]) / 50.0
        lorenz_y_n = float(lorenz_tuple[1]) / 50.0
        lorenz_z_n = float(lorenz_tuple[2]) / 50.0
        
        # 7. Biên dịch các trạng thái rời rạc và lỗi hệ thống (Flags & Indicators)
        decay_flag = 1.0 if float(math_state.get("decay_trigger", 0)) > 0 else 0.0
        
        # Tự động phát hiện sụp đổ số lượng thực thể vi mô
        any_population_collapse = 0.0
        if entity_count == 0 or prey_norm <= 0.001:
            any_population_collapse = 1.0
            
        # Thu thập số lượng vi phạm logic từ chu kỳ trước
        violations_dict = math_state.get("violations", {})
        raw_violation_count = float(len(violations_dict))

        # --- BƯỚC 3: ĐÓNG GÓI THÀNH VECTOR MATRIX CHUẨN CÔNG NGHIỆP ---
        feature_vector = np.array([
            e_ratio,                  # [0]
            m_ratio,                  # [1]
            prey_norm,                # [2]
            predator_norm,            # [3]
            v_max_norm,               # [4]
            e_burn_norm,              # [5]
            lorenz_x_n,               # [6]
            lorenz_y_n,               # [7]
            lorenz_z_n,               # [8]
            decay_flag,               # [9]
            any_population_collapse,  # [10]
            raw_violation_count       # [11]
        ], dtype=np.float32)

        # Ngăn chặn các lỗi toán học tràn số bí ẩn (NaN hoặc Inf Guard) bằng cách clipping
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_vector
    