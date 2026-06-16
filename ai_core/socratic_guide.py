# ai_core/socratic_guide.py

"""
ai_core/socratic_guide.py
========================================================

Socratic Guidance Engine

Vai trò:
- Sinh câu hỏi Socratic bằng Local LLM.
- Không giải đáp trực tiếp.
- Dẫn dắt người dùng tự phát hiện lỗi.
- Có cache + retry + anti-spam.
- Fallback nếu Local LLM chết.

Compatible:
- LM Studio
- OpenAI compatible endpoint
- Gemma
- Qwen
- Llama

Author:
Production Framework Edition
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

from typing import Dict, Any, Optional

import httpx

logger = logging.getLogger("SocraticGuide")


SYSTEM_PROMPT = """
Bạn là Socratic Tutor.

Nhiệm vụ:

1. KHÔNG đưa đáp án trực tiếp.
2. KHÔNG sửa lỗi hộ người dùng.
3. Chỉ đặt câu hỏi dẫn dắt.
4. Tối đa 2 câu hỏi.
5. Khuyến khích tư duy nhân quả.
6. Tập trung vào logic hệ kín.
7. Không dài dòng.

Luôn khiến người học tự suy luận.
"""


class SocraticGuide:

    def __init__(
        self,
        api_base: str = "http://192.168.1.46:1234/v1",
        model_name: str = "google/gemma-4-26b-a4b-qat",
    ):

        self.api_base = api_base
        self.model_name = model_name

        self.timeout = httpx.Timeout(
            timeout=30.0,
            connect=3.0
        )

        self._cache: Dict[str, str] = {}

        self._last_violation = None
        self._last_time = 0.0

        self.spam_window = 30

        self.static_templates = {

            "S1_energy_overflow":
                "Nếu tổng năng lượng tiếp tục tăng nhưng khả năng lưu trữ không đổi, thành phần nào của hệ kín sẽ chịu tác động đầu tiên?",

            "S2_mass_overflow":
                "Khi khối lượng vượt giới hạn thiết kế, bạn nghĩ điều gì xảy ra với cấu trúc không gian khả dụng?",

            "S3_prey_extinction":
                "Nếu loài săn mồi tồn tại nhưng nguồn thức ăn biến mất, điều gì sẽ xảy ra tiếp theo trong chuỗi sinh thái?",

            "S4_speed_violation":
                "Tốc độ hiện tại có đang vượt quá giới hạn mà các quy luật nền tảng cho phép hay không?",

            "default":
                "Bạn có chắc mọi giả định trong hệ kín hiện tại đều nhất quán với nhau không?"
        }

    # =====================================================
    # Utility
    # =====================================================

    def _make_cache_key(
        self,
        violation_type: str,
        context: Dict[str, Any]
    ) -> str:

        raw = violation_type + json.dumps(
            context,
            sort_keys=True,
            ensure_ascii=False
        )

        return hashlib.md5(
            raw.encode("utf-8")
        ).hexdigest()

    # =====================================================
    # Root Cause Extraction
    # =====================================================

    def _extract_root_cause(
        self,
        violation_type: str,
        context: Dict[str, Any]
    ) -> str:

        if violation_type == "S1_energy_overflow":

            current = context.get("current")
            limit = context.get("E_max")

            return (
                f"Current energy = {current}, "
                f"Maximum allowed = {limit}"
            )

        if violation_type == "S2_mass_overflow":

            current = context.get("current")
            limit = context.get("M_max")

            return (
                f"Current mass = {current}, "
                f"Maximum allowed = {limit}"
            )

        if violation_type == "S4_speed_violation":

            current = context.get("current")
            limit = context.get("V_max")

            return (
                f"Current speed = {current}, "
                f"Maximum allowed = {limit}"
            )

        return json.dumps(
            context,
            ensure_ascii=False
        )

    # =====================================================
    # Anti Spam
    # =====================================================

    def _is_spam(
        self,
        violation_type: str
    ) -> bool:

        now = time.time()

        if (
            self._last_violation == violation_type
            and
            now - self._last_time < self.spam_window
        ):
            return True

        self._last_violation = violation_type
        self._last_time = now

        return False

    # =====================================================
    # LLM Query
    # =====================================================

    async def _query_llm(
        self,
        violation_type: str,
        context: Dict[str, Any]
    ) -> str:

        root_cause = self._extract_root_cause(
            violation_type,
            context
        )

        user_prompt = f"""
Violation Type:
{violation_type}

Root Cause:
{root_cause}

Full Context:
{json.dumps(context, ensure_ascii=False, indent=2)}

Hãy đặt một câu hỏi Socratic ngắn gọn.
Không được giải thích đáp án.
"""

        payload = {

            "model": self.model_name,

            "messages": [

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 120
        }

        for attempt in range(3):

            try:

                async with httpx.AsyncClient(
                    timeout=self.timeout
                ) as client:

                    response = await client.post(
                        f"{self.api_base}/chat/completions",
                        json=payload
                    )

                    response.raise_for_status()

                    data = response.json()

                    return (
                        data["choices"][0]
                        ["message"]
                        ["content"]
                        .strip()
                    )

            except Exception as e:

                logger.warning(
                    f"[SocraticGuide] "
                    f"Attempt {attempt+1}/3 failed: {e}"
                )

                await asyncio.sleep(0.5)

        raise RuntimeError(
            "LLM unavailable"
        )

    # =====================================================
    # Main API
    # =====================================================

    async def guide(
        self,
        violation_type: str,
        context: Dict[str, Any]
    ) -> Optional[str]:

        logger.info(
            f"[SocraticGuide] {violation_type}"
        )

        if self._is_spam(
            violation_type
        ):
            return None

        cache_key = self._make_cache_key(
            violation_type,
            context
        )

        if cache_key in self._cache:

            return self._cache[
                cache_key
            ]

        try:

            result = await self._query_llm(
                violation_type,
                context
            )

            self._cache[
                cache_key
            ] = result

            return result

        except Exception as e:

            logger.error(
                "[SocraticGuide] "
                f"Fallback mode: {e}"
            )

            return self.static_templates.get(
                violation_type,
                self.static_templates["default"]
            )


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":

    async def run():

        guide = SocraticGuide()

        context = {

            "E_max": 1000,
            "current": 1500,

            "source":
                "Mana Generator Mk2",

            "storage":
                "Crystal Core"
        }

        result = await guide.guide(
            "S1_energy_overflow",
            context
        )

        print(result)

    asyncio.run(run())
