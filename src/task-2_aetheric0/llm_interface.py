from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"
    ERROR = "error"

class LLMResponse(BaseModel):
    status: ValidationStatus
    reason: str = Field(..., description="Algorithmic or semantic justificaition for the validation status.")

class BaseLLMValidator(ABC):
    @abstractmethod
    def evaluate_semantic_support(self, claim_text: str, evidence_text: str) -> LLMResponse:
        """
        Evaluate if the evidence semantically supports the claime.
        Must return a vlaidated LLMResponse object.
        """

class MockLLMValidator(BaseLLMValidator):
    """
    An offline mock for tests and local execution.
    Simulates semantic understanding without hitting external APIs.
    """
    def __init__(self, custom_rules: list[tuple[str, str, ValidationStatus, str]] | None):
        # Format: (evidence_substring, claim_substring, status, reason)
        self.custom_rules = custom_rules or []

    def evaluate_semantic_support(self, claim_text: str, evidence_text: str) -> LLMResponse:
        claim_lower = claim_text.lower()
        evidence_lower = evidence_text.lower()

        # Evaluate injected test rules first
        for ev_pat, claim_pat, status, reason in self.custom_rules:
            if ev_pat in evidence_lower and claim_pat in claim_lower:
                return LLMResponse(status=status, reason=reason)

        # Default fallback
        if "upward pressure" in evidence_lower and "increase costs" in claim_lower:
            return LLMResponse(
                status=ValidationStatus.SUPPORTED,
                reason="[Mock LLM] Semantic match: 'upward pressure' implies 'increase costs'"
            )
        
        return LLMResponse(
            status=ValidationStatus.UNCERTAIN,
            reason="[Mock LLM] Unable to verify semantic support with certainty."
        )

class RealLLMValidator(BaseLLMValidator):
    """
    Stub for a real LLM API call (e.g., Groq/OpenAI).
    To use this, you would implement the API request here.
    """
    def __init__(self, api_key: str, model_name: str="llama-3-70b-8192"):
        if not api_key:
            raise ValueError("RealLLMValidator requires a valid API key.")
        self.api_key = api_key
        self.model_name = model_name

    def evaluate_semantic_support(self, claim_text: str, evidence_text: str) -> LLMResponse:
        # Prevent silent None returns by raising NotImplementedError explicitly
        raise NotImplementedError(
            f"Real API execution with model '{self.model_name}' is disabled in offline mode."
        )


