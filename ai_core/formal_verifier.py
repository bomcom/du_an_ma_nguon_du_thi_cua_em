"""
Filename: utils/formal_verifier.py
Description: FORMAL_VERIFICATION_LAYER — Bộ kiểm chứng hình thức tiên đề.
             Kiểm tra tính nhất quán toán học (Mathematical Consistency) của hệ tiên đề S1...Sn
             trước khi cấp phát tài nguyên khởi chạy mô phỏng.
             Đóng vai trò như một Constraint Solver (Bộ giải ràng buộc) hạng nhẹ.
Author: Chuyên gia phần mềm AI/Simulation
"""

import logging
from typing import Dict, Any, Tuple, List


logger = logging.getLogger("FormalVerifier")

class FormalVerifier:
    """
    Công cụ chứng minh hình thức (Formal Verification Tool).
    Quét qua các hệ số cấu hình ban đầu (Initial State Configuration) do NLP_IDEA_PARSER sinh ra
    để phát hiện mâu thuẫn logic, dị điểm (Singularity), hoặc sự vi phạm định luật bảo toàn.
    """
    
    def __init__(self):
        logger.info("Khởi tạo FORMAL_VERIFICATION_LAYER. Sẵn sàng quét nghịch lý tiên đề.")
        
    def verify_initial_axioms(self, 
                              s1_config: Dict[str, float], 
                              s2_config: Dict[str, float], 
                              sn_custom_rules: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Thực hiện chứng minh hình thức trên tập hợp các biến số đầu vào.
        
        Args:
            s1_config: Tập hợp cấu hình Không gian/Năng lượng (Ví dụ: E_max, E_input).
            s2_config: Tập hợp cấu hình Sinh khối/Khối lượng (Ví dụ: M_max, M_input).
            sn_custom_rules: Danh sách các quy luật phụ do người dùng tự thiết lập.
            
        Returns:
            Tuple[bool, str]: (Đạt/Không Đạt, Chuỗi giải thích nguyên nhân gốc rễ nếu lỗi).
        """
        logger.info("[FormalVerifier] Đang tiến hành giải ràng buộc hệ tiên đề (Constraint Solving)...")

        # =====================================================================
        # BƯỚC 1: KIỂM CHỨNG ĐỊNH LUẬT BẢO TOÀN S_1 (ENERGY CONSERVATION)
        # =====================================================================
        total_energy_input = s1_config.get("total_energy_input", 0.0)
        e_max_capacity = s1_config.get("e_max_capacity", 1000.0)
        
        if e_max_capacity <= 0:
            return False, "Nghịch lý S1: Trần năng lượng (E_max) phải là một số dương lớn hơn 0."
            
        if total_energy_input > e_max_capacity:
            return False, (f"Vi phạm Bảo toàn S1: Năng lượng khởi tạo yêu cầu ({total_energy_input} J) "
                           f"vượt quá sức chứa tối đa của hệ kín ({e_max_capacity} J). "
                           "Hệ thống sẽ sụp đổ nhiệt ngay khi khởi chạy.")

        # =====================================================================
        # BƯỚC 2: KIỂM CHỨNG ĐỊNH LUẬT KHỐI LƯỢNG S_2 (MASS CEILING)
        # =====================================================================
        total_mass_input = s2_config.get("total_mass_input", 0.0)
        m_max_capacity = s2_config.get("m_max_capacity", 5000.0)
        
        if m_max_capacity <= 0:
            return False, "Nghịch lý S2: Giới hạn khối lượng (M_max) phải là một số dương lớn hơn 0."
            
        if total_mass_input > m_max_capacity:
            return False, (f"Vi phạm Trần Sinh khối S2: Khối lượng khởi tạo ({total_mass_input} kg) "
                           f"đã phá vỡ giới hạn không gian ({m_max_capacity} kg). "
                           "Dị điểm (Singularity) có thể hình thành.")

        # =====================================================================
        # BƯỚC 3: KIỂM CHỨNG ĐỘNG LỰC HỌC VÀ LỖI CHIA CHO 0 (ZERO-DIVISION GUARD)
        # =====================================================================
        gravity = s1_config.get("gravity", 9.81)
        friction_coeff = s1_config.get("friction", 0.1)

        if gravity <= 0:
            return False, "Nghịch lý Cơ học: Trọng lực phải là số dương."
        
        # Ví dụ: Nếu người dùng set lực cản môi trường (friction) âm -> vi phạm nhiệt động lực học
        if friction_coeff < 0:
            return False, "Nghịch lý Cơ học: Hệ số ma sát không thể âm. Điều này sẽ tạo ra năng lượng vĩnh cửu vô lý."

        # =====================================================================
        # BƯỚC 4: KIỂM CHỨNG QUY LUẬT TÙY BIẾN S_N (CUSTOM RULE CONFLICTS)
        # =====================================================================
        # Quét qua các rule do NLP_IDEA_PARSER dịch ra để tìm mâu thuẫn chéo
        rule_targets = set()
        for rule in sn_custom_rules:
            # Giả định rule có dạng: {"target_variable": "energy", "trend": "increase_infinite"}
            target = rule.get("target_variable")
            trend = rule.get("trend")
            
            if trend == "increase_infinite" and target == "energy":
                return False, "Mâu thuẫn Tiên đề: Bạn thiết lập quy luật 'Năng lượng tăng vô hạn', nhưng S1 lại có giới hạn E_max cố định. Hệ thống vô nghiệm."
                
            rule_targets.add(target)

        # Nếu vượt qua tất cả các chốt chặn hình thức
        logger.info("[FormalVerifier] Hệ tiên đề Nhất quán. Không phát hiện nghịch lý toán học.")
        return True, "Hệ thống đã sẵn sàng khởi tạo."

# =====================================================================
# HƯỚNG DẪN TÍCH HỢP VÀO MAIN.PY (Ngay trước khi tạo ECS và Threads)
# =====================================================================
"""
if __name__ == "__main__":
    from utils.formal_verifier import FormalVerifier
    
    verifier = FormalVerifier()
    
    # Giả lập dữ liệu do NLP_IDEA_PARSER bóc tách từ lệnh của người dùng
    s1_data = {"total_energy_input": 1500.0, "e_max_capacity": 1000.0, "gravity": 9.8}
    s2_data = {"total_mass_input": 100.0, "m_max_capacity": 5000.0}
    sn_rules = [{"target_variable": "mana", "trend": "decay"}]
    
    # Kiểm chứng TRƯỚC KHI cấp phát bộ nhớ cho thế giới
    is_valid, reason = verifier.verify_initial_axioms(s1_data, s2_data, sn_rules)
    
    if not is_valid:
        print(f"[FATAL ERROR] KHỞI TẠO BỊ CHẶN: {reason}")
        print("Vui lòng kích hoạt Socratic Guide để người dùng sửa lại tiên đề.")
        # sys.exit(1) hoặc gọi AI hướng dẫn
    else:
        # Nếu OK, mới gọi controller.run()
        pass
"""

