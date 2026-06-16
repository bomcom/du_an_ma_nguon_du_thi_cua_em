"""
input_parsers/nlp_idea_parser.py
================================
Natural Language to Deterministic ECS Component Compiler.

Architecture
------------
    Chịu trách nhiệm giao tiếp với Local SLM (Qwen, Gemma-GGUF) thông qua
    giao thức API tương thích OpenAI (LM Studio / Ollama).
    
    Quy trình (Pipeline):
        1. Nhận ý tưởng định tính (Semantic Prompt) từ người dùng.
        2. Ép buộc Local SLM biên dịch ý tưởng thành cấu trúc JSON nghiêm ngặt
           khớp 100% với hàm `inject_dynamic_component_from_schema_dict` của ECS.
        3. Gửi bản nháp JSON cho Adversarial Interrogator (Trọng tài Logic) để thẩm định.
        4. Nếu Trọng tài phê duyệt -> Bơm (Inject) trực tiếp vào ECSRegistry.
"""

import json
import logging
import asyncio
import aiohttp
import re
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("NLPIdeaParser")

class SemanticToMathCompiler:
    """
    Bộ biên dịch Ngôn ngữ tự nhiên sang Hệ Tiên đề Toán học.
    Vận hành hoàn toàn bất đồng bộ (Non-blocking) để không làm nghẽn luồng Lõi Toán.
    """

    def __init__(
        self, 
        ecs_registry: Any, 
        interrogator_gatekeeper: Callable[[Dict[str, Any]], bool],
        slm_api_url: str = "http://localhost:1234/v1/chat/completions",
        model_name: str = "qwen-2.5-7b-instruct",
        timeout_seconds: float = 15.0
    ):
        """
        Parameters
        ----------
        ecs_registry            : Tham chiếu đến đối tượng ECSRegistry.
        interrogator_gatekeeper : Hàm Callback tới ADVERSARIAL_INTERROGATOR_NET.
                                  Nhận JSON Schema, trả về True (Hợp lệ) / False (Vi phạm).
        slm_api_url             : URL của Local API (LM Studio mặc định chạy cổng 1234).
        """
        self.ecs_registry = ecs_registry
        self.gatekeeper = interrogator_gatekeeper
        self.api_url = slm_api_url
        self.model_name = model_name
        self.timeout = timeout_seconds

        # Prompt Hệ thống (System Prompt) - Cốt lõi để triệt tiêu Ảo giác Văn bản
        self._system_prompt = """
        You are a strict Data Compiler for a Deterministic ECS Engine.
        Your ONLY job is to convert the user's abstract physics/magic idea into a strict JSON schema.
        DO NOT output any conversational text, explanations, or markdown blocks. ONLY output raw JSON.

        The JSON must EXACTLY match this structure:
        {
            "component_name": "String (PascalCase, e.g., 'Mana', 'GravityCore')",
            "data_type": "float32",
            "mutability": "dynamic",
            "scope_binding": "entity",
            "attributes": {
                "key_name_1": 0.0,
                "key_name_2": 100.0
            }
        }
        Rule: All values inside "attributes" MUST be float numbers.
        """

    async def _query_local_slm(self, user_prompt: str) -> Optional[str]:
        """
        Gọi API Local SLM bất đồng bộ. Sử dụng định dạng tương thích OpenAI.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1, # Đặt temperature cực thấp để triệt tiêu sự sáng tạo ngẫu nhiên, ép tính logic
            "max_tokens": 300,
            "response_format": {"type": "json_object"} # Ép chuẩn JSON (nếu SLM hỗ trợ)
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, 
                    json=payload, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        logger.error(f"[NLPParser] SLM API trả về mã lỗi: {response.status}")
                        return None
                    
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                    
        except asyncio.TimeoutError:
            logger.error(f"[NLPParser] Local SLM phản hồi quá lâu (vượt quá {self.timeout}s).")
            return None
        except Exception as e:
            logger.error(f"[NLPParser] Lỗi kết nối đến Local SLM: {str(e)}")
            return None

    def _extract_and_validate_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Bóc tách JSON từ văn bản thô (phòng trường hợp SLM sinh ra markdown ```json)
        và xác thực cấu trúc cơ bản trước khi nạp vào hệ thống.
        """
        # Trích xuất đoạn text nằm giữa { và }
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            logger.error("[NLPParser] Không tìm thấy chuỗi JSON trong phản hồi của SLM.")
            return None

        json_str = match.group(0)
        try:
            schema_dict = json.loads(json_str)
            
            # Kiểm tra nhanh các key bắt buộc theo chuẩn của ECS
            required_keys = {"component_name", "data_type", "mutability", "scope_binding", "attributes"}
            if not required_keys.issubset(schema_dict.keys()):
                logger.error("[NLPParser] JSON bị thiếu các Key bắt buộc của ECS.")
                return None
                
            return schema_dict
        except json.JSONDecodeError:
            logger.error("[NLPParser] SLM sinh ra JSON không hợp lệ (Syntax Error).")
            return None

    async def compile_and_inject_idea(self, user_text: str) -> bool:
        """
        Quy trình chính (Main Pipeline):
        1. Gửi văn bản cho SLM.
        2. Chuyển đổi thành JSON Schema.
        3. Gửi cho Trọng tài Logic (Adversarial Interrogator) thẩm định.
        4. Nạp vào ECS nếu hợp lệ.
        """
        logger.info(f"[NLPParser] Bắt đầu biên dịch ý tưởng: '{user_text}'")
        
        # 1. Gọi SLM
        raw_response = await self._query_local_slm(user_text)
        if not raw_response:
            return False

        # 2. Bóc tách JSON
        schema_dict = self._extract_and_validate_json(raw_response)
        if not schema_dict:
            return False
            
        logger.info(f"[NLPParser] Dịch thành công Component: {schema_dict['component_name']}")

        # 3. Trọng tài thẩm định (Gatekeeper Check)
        # Truyền bản nháp schema_dict vào hàm callback của Adversarial Interrogator
        is_valid = self.gatekeeper(schema_dict)
        if not is_valid:
            logger.warning(f"[NLPParser] Trọng tài Logic đã TỪ CHỐI Component '{schema_dict['component_name']}' do vi phạm hệ tiên đề.")
            # Ở đây có thể kích hoạt Socratic Guide để phản hồi lại người dùng
            return False

        # 4. Găm động vào ECS
        injection_success = self.ecs_registry.inject_dynamic_component_from_schema_dict(schema_dict)
        if injection_success:
            logger.info(f"[NLPParser] >>> Đã GĂM (INJECT) Component '{schema_dict['component_name']}' vào Ma trận Thực tại!")
            return True
        else:
            logger.error("[NLPParser] Bơm Component thất bại tại tầng ECS.")
            return False

# =====================================================================
# HƯỚNG DẪN TÍCH HỢP (Mock implementation trong luồng chính)
# =====================================================================
"""
async def main():
    ecs = build_default_ecs()
    
    # Hàm wrap gọi đến Adversarial Interrogator
    def mock_gatekeeper_validation(schema: Dict) -> bool:
        # Ở đây bạn sẽ đưa data vào mạng Nơ-ron NumpyAdversarialNet
        # Tạm thời trả về True để test
        return True 

    parser = SemanticToMathCompiler(
        ecs_registry=ecs,
        interrogator_gatekeeper=mock_gatekeeper_validation,
        slm_api_url="http://localhost:1234/v1/chat/completions" # Chạy qua LM Studio
    )
    
    await parser.compile_and_inject_idea("Tạo một thuộc tính năng lượng ma thuật Mana, max 500, hồi phục 1.5")
"""
