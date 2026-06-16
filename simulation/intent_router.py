# intent_router.py

from enum import Enum
from dataclasses import dataclass
import re


class IntentType(Enum):
    COMPONENT = "component"
    ENTITY = "entity"
    WORLD = "world"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    original_prompt: str


class IntentRouter:
    """
    Phase 2 Router

    Chỉ làm nhiệm vụ:
        Prompt -> Intent

    KHÔNG spawn entity
    KHÔNG inject ECS
    KHÔNG gọi SLM

    => đúng SRP
    """

    COMPONENT_PATTERNS = [
        r"\bcomponent\b",
        r"\battribute\b",
        r"\bstat\b",
        r"\bmana\b",
        r"\bhealth\b",
        r"\benergy\b",
        r"thuộc tính",
        r"chỉ số",
        r"năng lượng",
    ]

    ENTITY_PATTERNS = [
        r"spawn",
        r"entity",
        r"creature",
        r"wolf",
        r"predator",
        r"prey",
        r"sinh ra",
        r"thêm thực thể",
        r"tạo con",
    ]

    WORLD_PATTERNS = [
        r"world",
        r"ecosystem",
        r"forest",
        r"desert",
        r"biome",
        r"thế giới",
        r"hệ sinh thái",
        r"khu rừng",
        r"sa mạc",
    ]

    def classify(self, prompt: str) -> IntentResult:

        text = prompt.lower()

        for pattern in self.COMPONENT_PATTERNS:
            if re.search(pattern, text):
                return IntentResult(
                    intent=IntentType.COMPONENT,
                    confidence=0.90,
                    original_prompt=prompt
                )

        for pattern in self.ENTITY_PATTERNS:
            if re.search(pattern, text):
                return IntentResult(
                    intent=IntentType.ENTITY,
                    confidence=0.90,
                    original_prompt=prompt
                )

        for pattern in self.WORLD_PATTERNS:
            if re.search(pattern, text):
                return IntentResult(
                    intent=IntentType.WORLD,
                    confidence=0.90,
                    original_prompt=prompt
                )

        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.0,
            original_prompt=prompt
        )
    