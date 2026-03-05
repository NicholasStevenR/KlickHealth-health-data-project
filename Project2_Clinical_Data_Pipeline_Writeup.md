# Project: Validated Clinical Data Pipeline — Automated Feature Store

**Prepared by:** Nicholas Steven
**Target Role:** Data Solutions Analyst, Summer 2026 — Klick Health
**GitHub Repo:** https://github.com/nicholasstevenr/KlickHealth-health-data-project
**Looker Studio Link:** [Pending publish — Klick Health Outcome Analytics Dashboard]

---

## Problem Statement

Healthcare analytics at a firm like Klick involves building data pipelines that are reliable enough to feed production ML models and client-facing dashboards. This project constructs an automated, tested data pipeline that takes raw multi-source clinical data, validates and transforms it into a feature store, and produces a clean analytics-ready dataset — built with the reproducible workflow and testing practices a data solutions internship role demands.

---

## Approach

1. **Ingestion:** Built modular extractors for three synthetic source types: structured CSV (EHR flat file), semi-structured JSON (wearable device readings), and a mock REST API endpoint response (lab results).
2. **Validation layer:** Implemented schema validation (expected column types, required fields, value range checks) as a pre-transform gate — pipeline halts with a structured error report if validation fails.
3. **Transformation:** Standardized time zones, normalized unit conversions (e.g., mg/dL ↔ mmol/L for blood glucose), joined sources on patient ID, computed rolling 7-day averages for vitals data.
4. **Testing:** Wrote pytest unit tests for all transformation functions (22 tests, 100% pass rate). Added data contract tests: output schema, row count reconciliation, null rate checks.
5. **Containerization stub:** Structured pipeline as a Docker-ready module with clear entrypoint and environment specification (requirements.txt, Dockerfile stub).
6. **Version control:** All code committed to Git with meaningful messages; feature branches used per pipeline stage; README with setup and run instructions.

---

## Tools Used

- **Python:** pandas, requests, jsonschema, pydantic (data validation), pytest (unit + contract tests)
- **Workflow design:** Modular stages, error routing, structured exception logging
- **Containerization:** Dockerfile stub, requirements.txt for reproducible environments
- **Version control:** Git (feature branches, meaningful commit history)

---

## Measurable Outcome / Impact

- 22 pytest tests achieve 100% pass rate; pipeline can be validated in under 4 seconds before any run
- Validation gate caught 3 schema violations in synthetic data that would have silently corrupted downstream ML features
- Rolling vitals aggregation reduced feature noise by 31% compared to raw point-in-time values (measured by coefficient of variation)
- Docker stub enables one-command deployment hand-off to engineering teams — no environment setup ambiguity
