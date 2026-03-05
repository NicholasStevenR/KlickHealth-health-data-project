"""
Healthcare Treatment Outcome Prediction — ML Pipeline
Author: Nicholas Steven
Target Role: Data Solutions Analyst, Summer 2026, Klick Health
Repo: github.com/nicholasstevenr/KlickHealth-health-data-project

End-to-end pipeline: synthetic EHR data → cleaned features → trained classifier → evaluation report.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# ── 1. Data loading ───────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["encounter_date", "followup_date"])


# ── 2. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Time-to-treatment (days from encounter to first treatment)
    df["days_to_treatment"] = (df["first_treatment_date"] - df["encounter_date"]).dt.days.clip(lower=0)

    # Comorbidity index (count of active chronic conditions)
    comorbidity_cols = ["has_diabetes", "has_hypertension", "has_copd", "has_chf", "has_ckd"]
    df["comorbidity_index"] = df[comorbidity_cols].sum(axis=1)

    # Medication adherence score proxy (fills / expected fills in prior 12 months)
    df["adherence_score"] = (df["medication_fills_12m"] / df["expected_fills_12m"].replace(0, np.nan)).clip(0, 1)
    df["adherence_score"] = df["adherence_score"].fillna(0)

    # Age group bins
    df["age_group"] = pd.cut(df["age_years"], bins=[0, 17, 34, 49, 64, 120],
                             labels=["0-17", "18-34", "35-49", "50-64", "65+"])

    # ED visits in prior year (utilization signal)
    df["high_utilizer"] = (df["ed_visits_12m"] >= 3).astype(int)

    return df


# ── 3. Preprocessing ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "age_years", "comorbidity_index", "adherence_score",
    "days_to_treatment", "ed_visits_12m", "high_utilizer",
    "lab_result_baseline", "bmi", "num_prior_hospitalizations",
    "specialist_referral", "primary_care_visits_12m", "polypharmacy_flag"
]
TARGET = "positive_outcome"

def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int)

    # Impute missing numeric values with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)

    # Address class imbalance with SMOTE
    smote = SMOTE(random_state=SEED, sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
    print(f"Class distribution after SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    return X_resampled, y_resampled, imputer


# ── 4. Model comparison ───────────────────────────────────────────────────────

def compare_models(X, y) -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=SEED),
        "Gradient Boosted Trees": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, random_state=SEED),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1")
        results[name] = {
            "auc_mean": auc_scores.mean().round(3),
            "auc_std":  auc_scores.std().round(3),
            "f1_mean":  f1_scores.mean().round(3),
        }
        print(f"{name}: AUC {auc_scores.mean():.3f} ± {auc_scores.std():.3f} | F1 {f1_scores.mean():.3f}")
    return results, models


# ── 5. Final model training and evaluation ────────────────────────────────────

def train_final(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=SEED)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\nFinal Model — Test AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No Improvement", "Positive Outcome"]))

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(fi.head(5).to_string())

    return model


# ── 6. Save and hand-off ──────────────────────────────────────────────────────

def save_model(model, imputer, outdir: str = "model_artifacts") -> None:
    Path(outdir).mkdir(exist_ok=True)
    joblib.dump(model,   f"{outdir}/outcome_model.pkl")
    joblib.dump(imputer, f"{outdir}/imputer.pkl")
    # Write hand-off spec
    spec = """# Model Hand-off Specification
## Input
- 12 numeric features (see FEATURE_COLS in pipeline)
- All numeric; imputer handles missing values
## Output
- predict(X) → binary label [0, 1]
- predict_proba(X)[:, 1] → probability of positive outcome
## Deployment notes
- Load via: joblib.load('outcome_model.pkl')
- Apply imputer FIRST, then model
- Threshold: 0.5 default; adjust for precision/recall trade-off per use case
## Retraining
- Re-run full pipeline when new data covers > 6-month window
- Validate AUC > 0.80 before deploying updated model
"""
    with open(f"{outdir}/handoff_spec.md", "w") as f:
        f.write(spec)
    print(f"Model artifacts saved to {outdir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df = load("data/patient_encounters_synthetic.csv")
    df = engineer_features(df)
    X, y, imputer = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    print("── Model Comparison (5-fold CV) ──")
    results, models = compare_models(X_train, y_train)

    print("\n── Final Model ──")
    best = train_final(X_train, y_train, X_test, y_test)
    save_model(best, imputer)
