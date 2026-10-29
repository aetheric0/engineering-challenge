import json

from llm_interface import MockLLMValidator
from models import Claim, Source
from validator import validate_claims


def run_example():
    with open("fixtures/data.json", "r") as f:
        data = json.load(f)

    sources = [Source(**s) for s in data["sources"]]
    claims = [Claim(**c) for c in data["claims"] if c["claim_id"] != "c6"]
    semantic_claim = [Claim(**c) for c in data["claims"] if c["claim_id"] == "c6"]

    print("--- Running Baseline (No LLM) ---")
    results = validate_claims(claims, sources)
    for r in results:
        print(f"[{r.claim_id}] {r.status.upper()}: {r.reason}")

    print("\n--- Running Semantic Claim with Mock LLM ---")
    llm = MockLLMValidator()
    semantic_result = validate_claims(semantic_claim, sources, llm=llm)[0]
    print(f"[{semantic_result.claim_id}] {semantic_result.status.upper()}: {semantic_result.reason}")

if __name__ == "__main__":
    run_example()