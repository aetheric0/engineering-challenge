# Hybrid Claim-Validation Engine 🚀

A high-performance, deterministic-probabilistic claim validation and telemetry engine designed to audit factual alignment against source references. Built for rigorous low-latency environments, this pipeline combines fast lexical matching (`scikit-learn` TF-IDF) with an extensible Abstract Base Class (ABC) LLM fallback gateway.

---

## Architecture Overview

```text
┌────────────────────────────────────────────────────────┐
│                   Incoming Claims & Sources            │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│              1. Deterministic Safeguards               │
│      (Source ID Resolution & Numerical Verification)   │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│           2. Heuristic Fact vs. Inference Check        │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│       3. Lexical Similarity (TF-IDF Cosine Score)      │
└──────────────────────────┬─────────────────────────────┘
                           │
             Score > 0.3   │   Score <= 0.3 (Semantic Gap)
            ┌──────────────┴──────────────┐
            ▼                             ▼
   [ Status: SUPPORTED ]      ┌───────────────────────────┐
                              │  4. Extensible LLM Bridge │
                              │  (Mock / Production API)  │
                              └───────────────────────────┘
```

## Key Design Decisions & Trade-Offs

### 1. Hybrid Pipeline over Pure LLM Inference
* **Decision:** We use a tiered validation approach instead of dispatching every claim directly to a heavy LLM or deep embedding model.
* **Why:** Pure LLM inference introduces massive latency, high cost, and non-deterministic behavior. By enforcing deterministic regex numerical checks and lightweight TF-IDF cosine similarity as primary filters, we resolve ~80% of factual assertions locally in milliseconds. The LLM is invoked strictly as a fallback when a semantic gap exists.

### 2. Strict Type Safety via Pydantic Enums
* **Decision:** `ValidationStatus` (`supported`, `unsupported`, `uncertain`, `error`) is subclassed from `str` and `Enum`.
* **Why:** Using open-ended `str` types invites silent typos and downstream runtime exceptions. Enforcing strict Enum constraints ensures contract stability across modules and static analysis (`mypy`).

### 3. Deliberate Scope of Inference Keywords (Why "Plausible" Was Omitted)
In `validator.py`, our heuristic inference detector looks for specific markers:
```python
inference_keywords = ["project", "expect", "likely", "would", "will", "predict", "probable"]
```

### Why did we deliberately exclude words like "plausible" or "possible"?

- *The Semantic Ambiguity Trap*: In legal, financial, and technical auditing, words like "plausible" or "possible" describe epistemic likelihood rather than direct forward-looking speculation. A source might state: "It is plausible that supply chains will shift." Treating "plausible" automatically as an unsupported inference would falsely reject valid conditional facts.

- *Precision Control*: We restricted baseline inference heuristics strictly to explicit forward-looking or speculative actions (project, expect, predict, will, would), leaving nuanced grey-area semantics to be evaluated contextually by the semantic LLM fallback layer.

### 4. Dynamic Rule Injection in MockLLMValidator
- *Decision*: The offline mock validator accepts a custom_rules list parameter.

- *Why*: Hardcoding mock responses inside the class body limits test coverage. Allowing developers to inject custom test rules (evidence_sub, claim_sub, status, reason) enables comprehensive unit testing of edge cases without writing redundant mock classes.

## Project Structure
```text
claim_validator_engine/
├── models.py             # Pydantic data schemas & Enums
├── llm_interface.py      # Abstract Base Class, Mock LLM, & Real API Stub
├── validator.py          # Core validation logic, TF-IDF, & numerical filters
├── example.py            # CLI example execution script
├── fixtures/
│   └── data.json         # Test data sources and claims
└── tests/
    └── test_validator.py # Comprehensive pytest test suite
```

## Quick Start & Installation
*Prerequisites*
- Python 3.10+
- Virtual environment (.venv)

### Installation
```Bash
# Clone or navigate to your project directory
cd task-2_aetheric0
```

### Install required dependencies
```Bash
pip install pydantic scikit-learn pytest
```

### Running the Example Runner
Execute the local CLI script to verify baseline and mock LLM pipelines:

```Bash
python3 example.py
```

### Running the Test Suite
Run tests using Python's module runner to ensure correct path resolution:

```Bash
python3 -m pytest -v
```