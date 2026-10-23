from pydantic import BaseModel, Field
from typing import Optional, Literal

class Source(BaseModel):
    source_id: str
    content: str
    locator: str

class Claim(BaseModel):
    claim_id: str
    text: str
    referenced_sources: list[str]

class EvidenceMatch(BaseModel):
    source_id: str
    locator: str
    excerpt: str

class ValidationResult(BaseModel):
    claim_id: str
    status: Literal["supported", "unsupported", "uncertain", "malformed"]
    classification: Literal["fact", "inference"]
    source_ids: list[str]
    evidence: list[EvidenceMatch]
    contradictions: list[str]
    unresolved_questions: list[str]
    reason: str
