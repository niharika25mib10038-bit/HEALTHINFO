# HEALTHINFO
# ==============================================================================
# HEART DISEASE PREDICTION — RESEARCH-READY ML PIPELINE V2
# XGBoost + Logistic Regression + Random Forest + SVC + Soft Voting Ensemble
# ==============================================================================

import os
import sys
import subprocess
import warnings
import json
import joblib

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. AUTO-INSTALL REQUIRED PACKAGES
# ==============================================================================

required_packages = {
    "xgboost": "xgboost",
    "shap": "shap",
    "joblib": "joblib"
}

for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"{package_name} not detected. Installing...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package_name
        ])

# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    brier_score_loss
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

# ==============================================================================
# 2. VISUALIZATION SETTINGS
# ==============================================================================

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size": 11,
    "figure.autolayout": True
})

# Create output directory
os.makedirs("outputs", exist_ok=True)

# ==============================================================================
# 3. DATASET
# ==============================================================================

print("\n" + "=" * 90)
print("HEART DISEASE MACHINE LEARNING PIPELINE")
print("=" * 90)

urls = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/heart-disease-cleveland.csv"
]

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

df = None

for url in urls:
    try:
        print(f"\nAttempting dataset source:")
        print(url)

        temp_df = pd.read_csv(
            url,
            header=None,
            names=columns
        )

        if len(temp_df) > 100:
            df = temp_df
            print("Dataset loaded successfully.")
            break

    except Exception as e:
        print("Source failed. Trying next source...")

if df is None:
    raise RuntimeError(
        "Could not load the Cleveland Heart Disease dataset."
    )

# ==============================================================================
# 4. DATA CLEANING
# ==============================================================================

print("\nCleaning dataset...")

df.replace("?", np.nan, inplace=True)

# Convert appropriate columns to numeric
for col in ["ca", "thal"]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Convert all remaining columns to numeric
for col in columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Convert target:
# 0 = no heart disease
# 1,2,3,4 = heart disease
df["target"] = (
    df["target"] > 0
).astype(int)

print(f"Dataset size: {df.shape[0]} rows × {df.shape[1]} columns")

print("\nTarget distribution:")
print(df["target"].value_counts())

# ==============================================================================
# 5. FEATURES AND TARGET
# ==============================================================================

X = df.drop(
    "target",
    axis=1
)

y = df["target"]

# ==============================================================================
# 6. TRAIN / TEST SPLIT
# ==============================================================================

# IMPORTANT:
# Split BEFORE preprocessing to prevent data leakage.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ==============================================================================
# 7. FEATURE GROUPS
# ==============================================================================

num_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

cat_features = [
    "cp",
    "restecg",
    "slope",
    "ca",
    "thal"
]

bin_features = [
    "sex",
    "fbs",
    "exang"
]

# ==============================================================================
# 8. PREPROCESSING
# ==============================================================================

num_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

cat_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

bin_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            num_pipeline,
            num_features
        ),
        (
            "cat",
            cat_pipeline,
            cat_features
        ),
        (
            "bin",
            bin_pipeline,
            bin_features
        )
    ]
)

# ==============================================================================
# 9. DEFINE MODELS
# ==============================================================================

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

lr = LogisticRegression(
    C=0.5,
    max_iter=1000,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

svc = SVC(
    C=1.0,
    kernel="rbf",
    probability=True,
    random_state=42
)

# ==============================================================================
# 10. MODEL PIPELINES
# ==============================================================================

models = {

    "XGBoost": xgb,

    "Logistic Regression": lr,

    "Random Forest": rf,

    "Support Vector Classifier": svc
}

pipelines = {}

for name, model in models.items():

    pipelines[name] = Pipeline([
        (
            "preprocessor",
            clone(preprocessor)
        ),
        (
            "model",
            clone(model)
        )
    ])

# ==============================================================================
# 11. SOFT-VOTING ENSEMBLE
# ==============================================================================

voting_clf = VotingClassifier(
    estimators=[
        ("xgb", clone(xgb)),
        ("lr", clone(lr)),
        ("rf", clone(rf)),
        ("svc", clone(svc))
    ],
    voting="soft"
)

ensemble_pipeline = Pipeline([
    (
        "preprocessor",
        clone(preprocessor)
    ),
    (
        "classifier",
        voting_clf
    )
])

pipelines["Soft Voting Ensemble"] = ensemble_pipeline

# ==============================================================================
# 12. 5-FOLD STRATIFIED CROSS-VALIDATION
# ==============================================================================

print("\n" + "=" * 90)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 90)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = {}

for name, pipeline in pipelines.items():

    print(f"\nEvaluating: {name}")

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results[name] = {
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Accuracy SD": scores["test_accuracy"].std(),

        "Precision Mean": scores["test_precision"].mean(),
        "Precision SD": scores["test_precision"].std(),

        "Recall Mean": scores["test_recall"].mean(),
        "Recall SD": scores["test_recall"].std(),

        "F1 Mean": scores["test_f1"].mean(),
        "F1 SD": scores["test_f1"].std(),

        "ROC-AUC Mean": scores["test_roc_auc"].mean(),
        "ROC-AUC SD": scores["test_roc_auc"].std()
    }

# ==============================================================================
# 13. CROSS-VALIDATION RESULTS TABLE
# ==============================================================================

cv_table = pd.DataFrame(cv_results).T

print("\n")
print("=" * 110)
print("TABLE I — 5-FOLD STRATIFIED CROSS-VALIDATION RESULTS")
print("=" * 110)

display_table = cv_table.copy()

for col in display_table.columns:
    display_table[col] = display_table[col].apply(
        lambda x: f"{x:.4f}"
    )

print(display_table.to_string())

cv_table.to_csv(
    "outputs/cross_validation_results.csv"
)

# ==============================================================================
# 14. FIT ALL MODELS ON TRAINING DATA
# ==============================================================================

print("\n" + "=" * 90)
print("HOLD-OUT TEST SET EVALUATION")
print("=" * 90)

test_results = {}

fitted_pipelines = {}

for name, pipeline in pipelines.items():

    print(f"\nFitting: {name}")

    pipeline.fit(
        X_train,
        y_train
    )

    fitted_pipelines[name] = pipeline

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    test_results[name] = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        ),

        "Brier Score": brier_score_loss(
            y_test,
            probabilities
        )
    }

# ==============================================================================
# 15. TEST RESULTS TABLE
# ==============================================================================

test_table = pd.DataFrame(
    test_results
).T

print("\n")
print("=" * 110)
print(
    f"TABLE II — HOLD-OUT TEST SET RESULTS "
    f"(N = {len(X_test)})"
)
print("=" * 110)

formatted_test_table = test_table.copy()

for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]:

    formatted_test_table[col] = formatted_test_table[
        col
    ].apply(
        lambda x: f"{x:.4f}"
    )

formatted_test_table["Brier Score"] = formatted_test_table[
    "Brier Score"
].apply(
    lambda x: f"{x:.4f}"
)

print(
    formatted_test_table.to_string()
)

test_table.to_csv(
    "outputs/test_set_results.csv"
)

# ==============================================================================
# 16. DETAILED CLASSIFICATION REPORT — ENSEMBLE
# ==============================================================================

ensemble_model = fitted_pipelines[
    "Soft Voting Ensemble"
]

ensemble_predictions = ensemble_model.predict(
    X_test
)

ensemble_probabilities = ensemble_model.predict_proba(
    X_test
)[:, 1]

print("\n")
print("=" * 90)
print("TABLE III — CLASSIFICATION REPORT: SOFT-VOTING ENSEMBLE")
print("=" * 90)

print(
    classification_report(
        y_test,
        ensemble_predictions,
        target_names=[
            "Healthy",
            "Heart Disease"
        ],
        digits=4,
        zero_division=0
    )
)

# ==============================================================================
# 17. CONFUSION MATRIX
# ==============================================================================

cm = confusion_matrix(
    y_test,
    ensemble_predictions
)

tn, fp, fn, tp = cm.ravel()

print("\n")
print("=" * 90)
print("TABLE IV — ENSEMBLE CONFUSION MATRIX")
print("=" * 90)

cm_table = pd.DataFrame(
    cm,
    index=[
        "Actual Healthy",
        "Actual Heart Disease"
    ],
    columns=[
        "Predicted Healthy",
        "Predicted Heart Disease"
    ]
)

print(cm_table)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

cm_table.to_csv(
    "outputs/confusion_matrix.csv"
)

# ==============================================================================
# 18. FIGURE 1 — CONFUSION MATRIX
# ==============================================================================

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=[
        "Healthy (0)",
        "Disease (1)"
    ],
    yticklabels=[
        "Healthy (0)",
        "Disease (1)"
    ]
)

plt.title(
    "Confusion Matrix — Soft Voting Ensemble",
    fontweight="bold"
)

plt.xlabel(
    "Predicted Class",
    fontweight="bold"
)

plt.ylabel(
    "Actual Class",
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

# ==============================================================================
# 19. FIGURE 2 — ROC CURVES FOR ALL MODELS
# ==============================================================================

plt.figure(
    figsize=(9, 7)
)

for name, pipeline in fitted_pipelines.items():

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    lw=2,
    label="Chance"
)

plt.xlabel(
    "False Positive Rate (1 - Specificity)",
    fontweight="bold"
)

plt.ylabel(
    "True Positive Rate (Sensitivity)",
    fontweight="bold"
)

plt.title(
    "ROC Curve Comparison",
    fontweight="bold"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/roc_curves.png",
    dpi=300
)

plt.show()

# ==============================================================================
# 20. CALIBRATION CURVE
# ==============================================================================

plt.figure(
    figsize=(8, 6)
)

for name, pipeline in fitted_pipelines.items():

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=5,
        strategy="uniform"
    )

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label=name
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    label="Perfect Calibration"
)

plt.xlabel(
    "Mean Predicted Probability",
    fontweight="bold"
)

plt.ylabel(
    "Fraction of Positives",
    fontweight="bold"
)

plt.title(
    "Model Probability Calibration",
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/calibration_curve.png",
    dpi=300
)

plt.show()

# ==============================================================================
# 21. XGBOOST FEATURE IMPORTANCE
# ==============================================================================

xgb_pipeline = fitted_pipelines[
    "XGBoost"
]

xgb_model = xgb_pipeline.named_steps[
    "model"
]

fitted_preprocessor = xgb_pipeline.named_steps[
    "preprocessor"
]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance_values = (
    xgb_model.feature_importances_
)

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

feature_importance_df[
    "Feature"
] = feature_importance_df[
    "Feature"
].str.replace(
    r"^[^_]+__",
    "",
    regex=True
)

feature_importance_df = (
    feature_importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n")
print("=" * 90)
print("TABLE V — XGBOOST FEATURE IMPORTANCE")
print("=" * 90)

print(
    feature_importance_df.to_string(
        index=False
    )
)

feature_importance_df.to_csv(
    "outputs/xgboost_feature_importance.csv",
    index=False
)

# ==============================================================================
# 22. FIGURE 4 — FEATURE IMPORTANCE
# ==============================================================================

plot_df = (
    feature_importance_df
    .sort_values(
        "Importance",
        ascending=True
    )
)

plt.figure(
    figsize=(9, 7)
)

plt.barh(
    plot_df["Feature"],
    plot_df["Importance"]
)

plt.xlabel(
    "XGBoost Feature Importance",
    fontweight="bold"
)

plt.ylabel(
    "Clinical Feature",
    fontweight="bold"
)

plt.title(
    "XGBoost Feature Importance",
    fontweight="bold"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/xgboost_feature_importance.png",
    dpi=300
)

plt.show()

# ==============================================================================
# 23. SHAP EXPLAINABILITY
# ==============================================================================

print("\n")
print("=" * 90)
print("SHAP EXPLAINABILITY")
print("=" * 90)

# Transform test data using the fitted preprocessing pipeline
X_test_transformed = fitted_preprocessor.transform(
    X_test
)

# SHAP TreeExplainer for XGBoost
explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = explainer.shap_values(
    X_test_transformed
)

# ------------------------------------------------------------------------------
# SHAP SUMMARY PLOT
# ------------------------------------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)

plt.title(
    "SHAP Feature Contribution — XGBoost"
)

plt.tight_layout()

plt.savefig(
    "outputs/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------------------------
# SHAP BAR PLOT
# ------------------------------------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title(
    "Mean Absolute SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "outputs/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==============================================================================
# 24. SAVE FINAL MODEL
# ==============================================================================

final_model_path = (
    "outputs/final_heart_disease_model.pkl"
)

joblib.dump(
    ensemble_model,
    final_model_path
)

print("\nFinal ensemble model saved to:")
print(final_model_path)

# ==============================================================================
# 25. SAVE MODEL METADATA
# ==============================================================================

metadata = {

    "project": "Heart Disease Prediction using Machine Learning",

    "dataset": "UCI Cleveland Heart Disease Dataset",

    "total_samples": int(len(df)),

    "training_samples": int(len(X_train)),

    "testing_samples": int(len(X_test)),

    "features": columns[:-1],

    "target": {
        "0": "No Heart Disease",
        "1": "Heart Disease"
    },

    "models": [
        "XGBoost",
        "Logistic Regression",
        "Random Forest",
        "Support Vector Classifier",
        "Soft Voting Ensemble"
    ],

    "ensemble_method": "Soft Voting",

    "cross_validation": {
        "method": "Stratified K-Fold",
        "folds": 5,
        "shuffle": True,
        "random_state": 42
    },

    "random_state": 42,

    "primary_model": "XGBoost",

    "explainability": [
        "XGBoost Feature Importance",
        "SHAP"
    ],

    "final_model": final_model_path
}

with open(
    "outputs/model_metadata.json",
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )

# ==============================================================================
# 26. SAVE COMPLETE SUMMARY
# ==============================================================================

summary = {

    "cross_validation": cv_table.to_dict(),

    "test_set": test_table.to_dict(),

    "confusion_matrix": {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

}

with open(
    "outputs/model_summary.json",
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

# ==============================================================================
# 27. FINAL SUMMARY
# ==============================================================================

print("\n")
print("=" * 90)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 90)

print("\nGenerated files:")

for filename in sorted(
    os.listdir("outputs")
):

    print(
        "  ✓",
        filename
    )

print("\nFinal model:")
print(
    "  outputs/final_heart_disease_model.pkl"
)

print("\nImportant:")
print(
    "This system is an educational/research ML model "
    "and must not be presented as a clinical diagnostic system."
)

print("\n" + "=" * 90)
