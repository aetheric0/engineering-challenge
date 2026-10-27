import re

from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_interface import BaseLLMValidator
from models import Claim, EvidenceMatch, Source, ValidationResult


def extract_numbers(text: str) -> list[str]:
    return re.findall(r'-?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b|-?\b\d+(?:\.\d+)?%?\b', text)

def is_inference(claim_text: str) -> bool:
    # Simple heuristic for the baseline: future tense or speculative words
    inference_keywords = ["project", "expect", "likely", "would", "will", "predict", "probable", "plausible"]
    return any(keyword in claim_text.lower() for keyword in inference_keywords)

def calculate_semantic_similarity(claim_text: str, source_text: str) -> float:
    """
    Computes TF-IDF cosine similarity between a claim and its referenced sources to handle minor paraphrasing without deep learning overhead.
    """
    if not claim_text.strip() or not source_text.strip():
        return 0.0
    try:
        corpus = [claim_text, source_text]
        vectorizer = TfidVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity_score)
    except ValueError:
        # Handles edge case where texts only contain stop words (empty vocabulary)
        return 0.0

def validate_claims(
        claims: list[Claim], 
        sources: list[Source],
        llm: BaseLLMValidator | None = None
    ) -> list[ValidationResult]:
    source_map = {s.source_id: s for s in sources}
    results = []

    for claim in claims:
        # 1. Resolve References (Missing sources)
        missing_sources = [s_id for s_id in claim.referenced_sources if s_id not in source_map]
        if missing_sources:
            results.append(
                ValidationResult(
                    claim_id=claim.claim_id,
                    status="malformed",
                    classification="fact",
                    source_ids=claim.referenced_sources,
                    evidence=[],
                    contradictions=[],
                    unresolved_questions=[f"Missing sources: {missing_sources}"],
                    reason="Claim references source IDs that do not exist in the provided evidence."
                )
            )
            continue

        # 2. Extract evidence (Take a larger snippet or search for keywords)
        valid_sources = [source_map[s_id] for s_id in claim.referenced_sources]
        evidence_matches = [
            EvidenceMatch(
                source_id=s.source_id,
                locator=s.locator,
                excerpt=s.content[:300] + "..." if len(s.content) > 300 else s.content
            )
            for s in valid_sources
        ]

        # 3. Check for Contradictions (Conflicting numerical evidence)
        all_source_text = " ".join([s.content for s in valid_sources])
        claim_numbers = [num.replace(",", "") for num in extract_numbers(claim.text)]
        source_numbers = [num.replace(",","") for num in extract_numbers(all_source_text)]

        contradictions = []
        unsupported_numbers = []

        if claim_numbers:
            for num in claim_numbers:
                # Distinguish between unsupported numbers vs contradictions if needed
                if num not in source_numbers:
                    unsupported_numbers.append(f"Claim number '{num}' not found in evidence.")

        # Check if multiple sources have different numbers for the same context
        if len(valid_sources) > 1 and len(source_numbers) > 1:
            unique_numbers = set(source_numbers)
            if len(unique_numbers) > 1 and not claim_numbers:
                contradictions.append(f"Supplied sources contain conflicting numbers: {unique_numbers}")

        # 4. Classify and Determine Status
        classification = "inference" if is_inference(claim.text) else "fact"

        if contradictions:
            status = "uncertain" if len(valid_sources) > 1 else "unsupported"
            reason = "Conflicting or missing numerical evidence prevents a definitive answer."
        elif classification == "inference":
            # if it's an inference, even if related, it's not a factual support
            status = "unsupported"
            reason = "The claim is an inference/speculation and is not directly stated as fact in the evidence."
        elif unsupported_numbers:
            status = "unsupported"
            reason = "The claim contains numerical elements not found in the source."
        else:
            # 5. Use TF-IDF Cosine Similarity
            similarity_score = calculate_semantic_similarity(claim.text, all_source_text)

            if similarity_score > 0.3:
                status = "supported"
                reason = f"TF-IDF cosine similarity ({similarity_score:.2f}) indicates direct support from the source."
            else:
                # Fallback to optional LLM if provided
                if llm:
                    llm_res = llm.evaluate_semantic_support(claim.text, all_source_text)
                    status= llm_res.status
                    reason = llm_res.reason
                else:
                    status = "unsupported"
                    reason = "The evidence does not provide adequate support for the claim."

        results.append(ValidationResult(
            claim_id=claim.claim_id,
            status=status,
            classification=classification,
            source_ids=claim.referenced_sources,
            evidence=evidence_matches,
            contradictions=contradictions,
            unresolved_questions=[f"Unsupported elements: {unsupported_numbers}"],
            reason=reason
        ))
    return results