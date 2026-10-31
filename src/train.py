
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import prepare_features, TARGET


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "product_returns.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def build_preprocessor(X):

    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python src/data_generation.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = prepare_features(df)
    y = df[TARGET].astype(int)

    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

    # Keep the test set separate until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Models and hyperparameter search spaces.
    candidates = {

        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),
            {
                "model__C": [0.01, 0.1, 1, 5, 10]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            {
                "model__n_estimators": [200, 350, 500],
                "model__max_depth": [None, 8, 12, 18],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            }
        ),

        "Extra Trees": (
            ExtraTreesClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            {
                "model__n_estimators": [200, 350, 500],
                "model__max_depth": [None, 8, 12, 18],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            }
        ),

        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                random_state=42
            ),
            {
                "model__max_iter": [100, 180, 250],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_leaf_nodes": [15, 25, 31],
                "model__l2_regularization": [0.0, 0.5, 1.0]
            }
        )
    }

    results = []
    fitted_models = {}

    for name, (estimator, parameters) in candidates.items():

        print(f"\nTraining and tuning: {name}")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameters,
            n_iter=8,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_

        # Evaluate only after model selection.
        probability = best_pipeline.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)

        metrics = {
            "model": name,
            "cv_roc_auc": search.best_score_,
            "roc_auc": roc_auc_score(y_test, probability),
            "average_precision": average_precision_score(
                y_test, probability
            ),
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(
                y_test, prediction, zero_division=0
            ),
            "recall": recall_score(
                y_test, prediction, zero_division=0
            ),
            "f1": f1_score(
                y_test, prediction, zero_division=0
            )
        }

        results.append(metrics)
        fitted_models[name] = best_pipeline

        print("Best CV ROC-AUC:", round(search.best_score_, 4))
        print("Test ROC-AUC:", round(metrics["roc_auc"], 4))
        print("Best parameters:", search.best_params_)

    # Select the model using cross-validation, not test ROC-AUC.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "cv_roc_auc",
        ascending=False
    )

    best_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_models[best_name]

    best_probability = best_pipeline.predict_proba(X_test)[:, 1]
    best_prediction = (best_probability >= 0.5).astype(int)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_pipeline,
        MODEL_DIR / "return_risk_pipeline.joblib"
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    report = classification_report(
        y_test,
        best_prediction,
        output_dict=True,
        zero_division=0
    )

    with open(
        REPORT_DIR / "classification_report.json",
        "w"
    ) as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(
        confusion_matrix(y_test, best_prediction),
        index=["actual_no_return", "actual_return"],
        columns=["predicted_no_return", "predicted_return"]
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    metadata = {
        "best_model": best_name,
        "cv_roc_auc": float(results_df.iloc[0]["cv_roc_auc"]),
        "test_roc_auc": float(
            roc_auc_score(y_test, best_probability)
        ),
        "test_average_precision": float(
            average_precision_score(y_test, best_probability)
        ),
        "threshold": 0.5,
        "note": (
            "Synthetic demonstration data; "
            "not validated on real transactions."
        )
    }

    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    print("\nBest model:", best_name)
    print("Final test ROC-AUC:", round(
        roc_auc_score(y_test, best_probability), 4
    ))
    print("\nModel and reports saved successfully.")


if __name__ == "__main__":
    main()
