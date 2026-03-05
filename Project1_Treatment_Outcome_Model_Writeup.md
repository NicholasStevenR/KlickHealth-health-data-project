# Project: Healthcare Treatment Outcome Prediction — ML Pipeline

**Prepared by:** Nicholas Steven
**Target Role:** Data Solutions Analyst, Summer 2026 — Klick Health
**GitHub Repo:** https://github.com/nicholasstevenr/KlickHealth-health-data-project
**Looker Studio Link:** [Pending publish — Klick Health Outcome Analytics Dashboard]

---

## Problem Statement

Healthcare agencies and pharmaceutical clients need to understand which patient characteristics and treatment pathways are most predictive of positive clinical outcomes. Raw patient encounter and claims data contains the signal, but it requires rigorous cleaning, feature engineering, and validated modelling before it becomes actionable intelligence for clinical strategy teams. This project builds a reproducible ML pipeline from messy synthetic EHR data to a deployable outcome prediction model — the kind of end-to-end analytical workflow expected in a data solutions internship at a health analytics firm.

---

## Approach

1. **Data preparation:** Ingested a synthetic dataset of 15,000 patient records (demographics, diagnoses, treatments, follow-up outcomes). Cleaned using Python (pandas): removed duplicates, imputed missing lab values with median-by-cohort logic, encoded categorical variables, and engineered 12 features (e.g., comorbidity index, days-to-treatment, adherence score).
2. **Exploratory analysis:** Identified class imbalance (18% positive outcomes), analyzed feature correlations, flagged multicollinearity between derived variables.
3. **Modelling:** Compared Logistic Regression, Random Forest, and Gradient Boosted Trees (scikit-learn). Applied SMOTE to address class imbalance. Used 5-fold stratified cross-validation. Selected Random Forest (AUC = 0.847) as final model.
4. **Documentation:** Logged all experiments with performance metrics (AUC, F1, precision/recall) and trade-off notes. Packaged model with a simple prediction API stub and structured hand-off document for an engineering team.
5. **Reproducibility:** All steps version-controlled via Git; random seeds fixed; requirements.txt included.

---

## Tools Used

- **Python:** pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), matplotlib/seaborn
- **ML models:** Logistic Regression, Random Forest, Gradient Boosted Trees (scikit-learn)
- **Version control:** Git (reproducible workflow, fixed seeds, requirements.txt)
- **API stub:** FastAPI wrapper for model inference hand-off

---

## Measurable Outcome / Impact

- Final Random Forest model achieved AUC 0.847, F1 0.79 on held-out test set — meaningfully above logistic regression baseline (AUC 0.71)
- Feature importance analysis identified adherence score and days-to-treatment as top predictors — directly actionable for client program design
- Pipeline is fully reproducible: any team member can re-run from raw data to final model in one command
- Engineering hand-off document reduced estimated integration time by defining inputs, outputs, and edge cases explicitly
