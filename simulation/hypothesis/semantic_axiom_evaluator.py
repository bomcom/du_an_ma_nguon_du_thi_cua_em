"""
Filename: simulation/hypothesis/semantic_axiom_evaluator.py
Description: SEMANTIC_AXIOM_EVALUATOR — Bộ định giá và thẩm định ngữ nghĩa tiên đề.
             Đánh giá chi phí (Energy, Mass, Entropy, Complexity) và 
             rủi ro (Replication, Ecosystem Disruption) của các thuộc tính tự do.
Author: Chuyên gia phần mềm AI/Simulation
"""

import logging
import math
from typing import Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger("SemanticAxiomEvaluator")

# =========================================================================
# 1. DATA MODELS (Đầu ra của quá trình thẩm định)
# =========================================================================

class SemanticEvaluation(BaseModel):
    """Bản báo cáo thẩm định chi tiết cho một Tiên đề (Hypothesis)."""
    approved: bool = False
    
    # Các loại "Thuế" phải trả để tồn tại
    energy_cost: float = Field(default=0.0, ge=0.0)
    mass_cost: float = Field(default=0.0, ge=0.0)
    entropy_cost: float = Field(default=0.0, ge=0.0)
    complexity_cost: float = Field(default=0.0, ge=0.0)
    
    # Các chỉ số rủi ro hệ thống (Thang điểm 0.0 -> 1.0)
    replication_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    ecosystem_disruption_score: float = Field(default=0.0, ge=0.0, le=1.0)
    axiom_stability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    
    rejection_reason: str = ""

# =========================================================================
# 2. CORE EVALUATOR (Bộ máy thẩm định)
# =========================================================================

class SemanticAxiomEvaluator:
    """
    Đánh giá một CandidateHypothesis dựa trên các phương pháp Heuristic toán học.
    Không dùng Whitelist. Phân tích trực tiếp dựa trên cấu trúc dữ liệu đề xuất.
    """
    
    def __init__(self, 
                 base_entropy_tax: float = 0.01, 
                 max_allowed_risk: float = 0.85):
        self.base_entropy_tax = base_entropy_tax
        self.max_allowed_risk = max_allowed_risk
        
        # Từ khóa mang rủi ro nhân bản/lạm phát cao
        self.high_risk_keywords = ["regen", "multiply", "spawn", "rate", "infinite", "growth", "auto"]
        logger.info("Khởi tạo SEMANTIC_AXIOM_EVALUATOR. Sẵn sàng định giá tiên đề mới.")

    def evaluate(self, concept_name: str, proposed_attributes: Dict[str, float]) -> SemanticEvaluation:
        """
        Thực hiện đánh giá toàn diện một tập hợp các thuộc tính động.
        """
        logger.info(f"[AxiomEvaluator] Đang thẩm định khái niệm: '{concept_name}'...")
        eval_result = SemanticEvaluation()
        
        if not proposed_attributes:
            eval_result.rejection_reason = "Khái niệm rỗng, không chứa thuộc tính nào để định giá."
            return eval_result

        # 1. Tính toán Complexity Cost (Chi phí phức tạp tính toán)
        # Hệ thống càng nhiều thuộc tính, độ phức tạp O(N^2) càng tăng
        N = len(proposed_attributes)
        eval_result.complexity_cost = (N ** 1.5) * 0.5 

        # 2. Quét Nội Suy (Heuristic Scanning) để tính Energy, Mass và Risk
        total_magnitude = 0.0
        risk_multipliers = 0
        
        for key, value in proposed_attributes.items():
            key_lower = key.lower()
            abs_val = abs(value)
            total_magnitude += abs_val
            
            # Ước lượng Mass/Energy dựa trên "tên" thuộc tính (Semantic Guessing)
            if any(k in key_lower for k in ["mass", "weight", "heavy", "density", "cap", "max"]):
                eval_result.mass_cost += abs_val * 0.1
            if any(k in key_lower for k in ["energy", "mana", "heat", "speed", "power", "force"]):
                eval_result.energy_cost += abs_val * 0.15
                
            # Đánh giá Replication Risk (Rủi ro sinh sôi/lạm phát)
            if any(risk_word in key_lower for risk_word in self.high_risk_keywords):
                risk_multipliers += 1
                eval_result.energy_cost += abs_val * 0.5  # Phạt thuế năng lượng nặng cho các thuộc tính tự sinh

        # 3. Tính toán Entropy (Sự hỗn loạn)
        # Dựa trên phương sai (Variance) hoặc độ lớn của các giá trị.
        eval_result.entropy_cost = self.base_entropy_tax * math.sqrt(total_magnitude) + eval_result.complexity_cost
        
        # 4. Tính toán rủi ro tổng thể (Ecosystem Disruption & Replication Risk)
        # Giới hạn risk score từ 0.0 đến 1.0 bằng hàm sigmoid: S(x) = 1 / (1 + e^-x)
        raw_risk = (risk_multipliers * 2.0) + (total_magnitude / 1000.0)
        eval_result.replication_risk = self._sigmoid_normalize(raw_risk)
        
        # Disruption Score dựa trên tỷ lệ Entropy sinh ra
        eval_result.ecosystem_disruption_score = min(1.0, eval_result.entropy_cost / 100.0)

        # 5. Tính Stability Score (Điểm Ổn định Tiên đề)
        # Càng rủi ro, càng phức tạp -> Càng kém ổn định
        eval_result.axiom_stability_score = max(0.0, 1.0 - (eval_result.replication_risk * 0.5 + eval_result.ecosystem_disruption_score * 0.5))

        # 6. RA QUYẾT ĐỊNH (APPROVAL LOGIC)
        if eval_result.ecosystem_disruption_score > self.max_allowed_risk:
            eval_result.approved = False
            eval_result.rejection_reason = f"Rủi ro sụp đổ hệ sinh thái quá cao ({eval_result.ecosystem_disruption_score:.2f} > {self.max_allowed_risk})."
        elif eval_result.axiom_stability_score < 0.2:
            eval_result.approved = False
            eval_result.rejection_reason = "Tiên đề quá thiếu ổn định. Cấu trúc nội tại sẽ tự sụp đổ."
        else:
            eval_result.approved = True

        logger.info(f"[AxiomEvaluator] Hoàn tất. Approved: {eval_result.approved} | Stability: {eval_result.axiom_stability_score:.2f}")
        return eval_result

    def _sigmoid_normalize(self, x: float) -> float:
        """Chuẩn hóa một giá trị bất kỳ về khoảng [0, 1]."""
        return 1 / (1 + math.exp(-x))

# =========================================================================
# TEST/DEMO MODULE NÀY ĐỘC LẬP
# =========================================================================
if __name__ == "__main__":
    evaluator = SemanticAxiomEvaluator()
    
    # Test Case 1: Một khái niệm bình thường (An toàn)
    normal_concept = {
        "max_health": 100.0,
        "armor": 5.0
    }
    res1 = evaluator.evaluate("BasicEntity", normal_concept)
    print("\n--- TEST 1: BasicEntity ---")
    print(res1.model_dump_json(indent=2))

    # Test Case 2: Một khái niệm Op/Nguy hiểm (Chứa rate, regen, số cực lớn)
    op_concept = {
        "mana_capacity": 50000.0,
        "mana_regen_rate": 1000.0,
        "infinite_growth_multiplier": 5.0
    }
    res2 = evaluator.evaluate("GodModeEntity", op_concept)
    print("\n--- TEST 2: GodModeEntity ---")
    print(res2.model_dump_json(indent=2))
    