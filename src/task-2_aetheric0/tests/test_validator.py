import pytest
from models import Source, Claim
from validator import validate_claims

from llm_interface import ValidationStatus

@pytest.fixture
def mock_data():
    sources = [
        Source(
            source_id="s1",
            content="The regulation was published on 1 June.",
            locator="p1"
        ),
        Source(
            source_id="s2",
            content="Market analysis shows cloud adoption is rising",
            locator="p2"
        ),
        Source(
            source_id="s3",
            content="Costs increased by 40%",
            locator="p3"
        )
    ]
    return sources

def test_direct_support(mock_data):
    claim = Claim(
        claim_id="c1",
        text="The regulation was published on 1 June.",
        referenced_sources=["s1"]
    )
    result = validate_claims([claim], mock_data)[0]
    assert result.status == ValidationStatus.SUPPORTED
    assert result.source_ids == ["s1"]
    assert len(result.evidence) == 1

def test_unsupported_numerical_claim(mock_data):
    claim = Claim(
        claim_id="c2",
        text="Costs increased by 30%",
        referenced_sources=["s1"]
    )
    result = validate_claims([claim], mock_data)[0]
    assert result.status == ValidationStatus.UNSUPPORTED
    assert "not found in evidence" in result.unresolved_questions[0]

def test_conflicting_excerpts(mock_data):
    # s1 says 10% implicitly (no wait, s1 says 1 June), let's use s3 vs a new s4
    sources = mock_data + [Source(
        source_id="s4",
        content="Content increased by 10%.",
        locator="p4"
        )]
    claim = Claim(
        claim_id="c4",
        text="Costs increased this year.",
        referenced_sources=["s3", "s4"]
    )
    result = validate_claims([claim], sources)[0]
    assert result.status == ValidationStatus.UNCERTAIN
    assert len(result.contradictions) > 0
    assert "conflicting numbers" in result.contradictions[0]

def test_inference_vs_fact(mock_data):
    claim = Claim(
        claim_id="c5",
        text="This will significantly increase costs.",
        referenced_sources=["s2"]
    )
    result = validate_claims([claim], mock_data)[0]
    assert result.classification == "inference"
    assert result.status == ValidationStatus.UNSUPPORTED
    assert "inference" in result.reason


def test_llm_semantic_fallback(mock_data):
    # We need the s2 source for this past
    sources = [
        s if s.source_id != "s2" else Source(
            source_id="s2",
            content="Projections indicate signficant upward pressure on clou spending.",
            locator="p2"
        ) for s in mock_data
    ]

    # This claim has almost zero keyword overlap with s2
    claim = Claim(claim_id="c6", text="Cloud costs are rising.", referenced_sources=["s2"])

    # 1. Without LLM, it should fail the baseline
    result_baseline = validate_claims([claim], sources)[0]
    assert result_baseline.status == ValidationStatus.UNSUPPORTED

    # 2. With Mock LLM, it should pass via semantic check
    from llm_interface import MockLLMValidator
    mock_llm = MockLLMValidator()
    result_llm = validate_claims([claim], sources, llm=mock_llm)[0]

    assert result_llm.status == ValidationStatus.SUPPORTED
    assert "[Mock LLM]" in result_llm.reason