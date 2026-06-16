"""
formal_hypothesis_compiler.py

Meta-Simulation Engine - Phase: Hypothesis-to-Axiom Pipeline
Architectural Standard: Semantic Firewall 4-Layers & Strict Lifecycle Management
"""

import json
import logging
import asyncio
import aiohttp
import re
from typing import Any, Dict, Optional, List
from enum import Enum
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("FormalHypothesisCompiler")

# =========================================================================
# 1. HYPOTHESIS LIFECYCLE (Strict State Machine)
# =========================================================================

class HypothesisStatus(str, Enum):
    DRAFT = "DRAFT"             # Vừa nhận từ User
    CANDIDATE = "CANDIDATE"     # Đã parse thành JSON & Pydantic hợp lệ (Tầng 1)
    VERIFIED = "VERIFIED"       # Vượt qua xác minh Toán học S1/S2 (Tầng 2)
    APPROVED = "APPROVED"       # Vượt qua Adversarial (Tầng 3) & Emergence (Tầng 4)
    ACTIVE = "ACTIVE"           # Đang được World Kernel mô phỏng
    DEPRECATED = "DEPRECATED"   # Bị phế truất do xung đột hệ thống sau này

# =========================================================================
# 2. STRICT HYPOTHESIS MODELS (Tầng 1: Schema Validation)
# =========================================================================

class CandidateHypothesis(BaseModel):
    """Cấu trúc giả thuyết nghiêm ngặt, chặn mọi thuộc tính lậu."""
    hypothesis_id: str = Field(default_factory=lambda: f"hyp_{__import__('uuid').uuid4().hex[:8]}")
    concept_name: str = Field(..., pattern=r"^[A-Z][a-zA-Z0-9]*$")
    description: str
    proposed_attributes: Dict[str, float]
    energy_cost_estimate: float = Field(default=0.0, ge=0.0)
    mass_cost_estimate: float = Field(default=0.0, ge=0.0)
    
    status: HypothesisStatus = HypothesisStatus.DRAFT
    validation_notes: List[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"

class ApprovedAxiom(BaseModel):
    """Tiên đề chân lý cuối cùng được đẩy vào World Kernel."""
    axiom_id: str
    concept_name: str
    description: str
    attributes: Dict[str, float]
    energy_cost: float
    mass_cost: float
    approved_tick: int
    stability_score: float = Field(default=1.0, le=1.0, ge=0.0)

# =========================================================================
# 3. AXIOM REGISTRY (Kho Lưu Trữ Chân Lý)
# =========================================================================

class AxiomRegistry:
    """Nơi duy nhất World Kernel truy xuất để cập nhật thế giới."""
    def __init__(self):
        self._axioms: Dict[str, ApprovedAxiom] = {}

    def register_axiom(self, axiom: ApprovedAxiom) -> bool:
        if axiom.axiom_id in self._axioms:
            logger.warning(f"Tiên đề {axiom.axiom_id} đã tồn tại.")
            return False
        self._axioms[axiom.axiom_id] = axiom
        logger.info(f"[Registry] Đã khắc ghi tiên đề: {axiom.concept_name}")
        return True

    def get_all_active(self) -> List[ApprovedAxiom]:
        return list(self._axioms.values())

# =========================================================================
# 4. FORMAL HYPOTHESIS COMPILER (Core Engine)
# =========================================================================

class FormalHypothesisCompiler:
    """
    Biên dịch ý tưởng ngôn ngữ tự nhiên thành Tiên đề.
    Tuyệt đối KHÔNG inject trực tiếp vào ECS.
    """

    def __init__(
        self,
        axiom_registry: AxiomRegistry,
        formal_verifier: Any,          # Tầng 2: Kiểm tra định luật bảo toàn
        adversarial_interrogator: Any, # Tầng 3: Kiểm tra nghịch lý, lặp vô hạn
        emergence_analyzer: Any,       # Tầng 4: Đánh giá tác động hệ sinh thái (Grok thiếu)
        slm_api_url: str = "http://localhost:1234/v1/chat/completions",
        model_name: str = "qwen-2.5-7b-instruct",
        timeout: float = 20.0
    ):
        self.registry = axiom_registry
        self.formal_verifier = formal_verifier
        self.interrogator = adversarial_interrogator
        self.emergence_analyzer = emergence_analyzer
        self.api_url = slm_api_url
        self.model_name = model_name
        self.timeout = timeout

        self._system_prompt = """
        You are a Hypothesis Formalizer for a deterministic simulation engine.
        Extract the user's concept into a strict JSON object. No explanations.
        {
          "concept_name": "PascalCase",
          "description": "Clear description",
          "proposed_attributes": {"prop_1": 10.0, "prop_2": 2.5},
          "energy_cost_estimate": 15.0,
          "mass_cost_estimate": 0.0
        }
        """

    async def _query_local_slm(self, user_prompt: str) -> Optional[str]:
        # Tương tự Grok, giữ nguyên logic gọi API tốt này
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            "response_format": {"type": "json_object"}
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"[SLM Error] {e}")
        return None

    def _parse_hypothesis(self, raw_text: str) -> Optional[CandidateHypothesis]:
        """TẦNG 1: Schema Validation thông qua Pydantic."""
        try:
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            json_str = match.group(1) if match else raw_text
            data = json.loads(json_str)
            
            # Khởi tạo giả thuyết, trạng thái mặc định là DRAFT
            hypothesis = CandidateHypothesis(**data)
            
            # Vượt qua Pydantic => Nâng cấp lên CANDIDATE
            hypothesis.status = HypothesisStatus.CANDIDATE
            return hypothesis
        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"[Tầng 1 Failed] Dữ liệu vi phạm Schema: {e}")
            return None

    async def compile_hypothesis(self, user_text: str, current_tick: int) -> Optional[ApprovedAxiom]:
        """
        Đường ống Semantic Firewall 4 Tầng.
        """
        logger.info(f"== BẮT ĐẦU BIÊN DỊCH Ý TƯỞNG ==")
        
        # 1. Trích xuất NLP -> JSON (TẦNG 1)
        raw_response = await self._query_local_slm(user_text)
        if not raw_response: return None
        
        hypothesis = self._parse_hypothesis(raw_response)
        if not hypothesis: return None
        logger.info(f"[Tầng 1 - OK] Candidate tạo thành công: {hypothesis.concept_name}")

        # 2. TẦNG 2: Formal Verification (Toán học & Bảo toàn S1/S2)
        # SỬA LỖI GROK: Truyền toàn bộ proposed_attributes động, không hardcode "max_capacity"
        is_formally_valid, reason = self.formal_verifier.verify_initial_axioms(
            proposed_state=hypothesis.proposed_attributes, 
            s2_config={"allow_energy_deficit": False}, # Ví dụ truyền config chuẩn
            custom_rules=[]
        )
        if not is_formally_valid:
            logger.warning(f"[Tầng 2 - Failed] Vi phạm bảo toàn: {reason}")
            return None
        
        hypothesis.status = HypothesisStatus.VERIFIED
        hypothesis.validation_notes.append("Đạt chuẩn bảo toàn S1/S2")
        logger.info("[Tầng 2 - OK] Xác minh toán học thành công.")

        # 3. TẦNG 3: Adversarial Challenge (Chất vấn nghịch lý/Hack hệ thống)
        interrogation_result = self.interrogator.interrogate({
            "concept": hypothesis.concept_name,
            "attributes": hypothesis.proposed_attributes
        })
        if interrogation_result.get("is_violation"):
            logger.warning(f"[Tầng 3 - Failed] Phát hiện rủi ro: {interrogation_result.get('rca')}")
            return None
        logger.info("[Tầng 3 - OK] Không phát hiện vòng lặp vô hạn.")

        # 4. TẦNG 4: Emergence Impact Analysis (Đánh giá Hệ sinh thái)
        # BỔ SUNG LỖI THIẾU CỦA GROK
        impact_score = self.emergence_analyzer.analyze_impact(hypothesis.proposed_attributes)
        if impact_score > 0.9: # Ví dụ: Ngưỡng phá vỡ cân bằng là 0.9
            logger.warning(f"[Tầng 4 - Failed] Ý tưởng quá overpowered (Score: {impact_score}). Từ chối.")
            return None
        logger.info(f"[Tầng 4 - OK] Tác động hệ sinh thái an toàn (Score: {impact_score}).")

        # =====================================================================
        # PHÊ DUYỆT CUỐI CÙNG (FINAL APPROVAL)
        # =====================================================================
        hypothesis.status = HypothesisStatus.APPROVED
        
        approved_axiom = ApprovedAxiom(
            axiom_id=f"ax_{hypothesis.hypothesis_id}",
            concept_name=hypothesis.concept_name,
            description=hypothesis.description,
            attributes=hypothesis.proposed_attributes,
            energy_cost=hypothesis.energy_cost_estimate,
            mass_cost=hypothesis.mass_cost_estimate,
            approved_tick=current_tick,
            stability_score=(1.0 - impact_score) # Điểm ổn định tỷ lệ nghịch với độ biến động
        )

        # Lưu vào Cuốn sách chân lý
        self.registry.register_axiom(approved_axiom)
        logger.info(f"== BIÊN DỊCH HOÀN TẤT: {approved_axiom.concept_name} ==")
        
        return approved_axiom
    