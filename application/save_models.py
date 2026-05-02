from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from interpret.glassbox import ExplainableBoostingRegressor


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "CSV_data"
MODEL_DIR = Path(__file__).resolve().parent / "app_models"

MODEL_DIR.mkdir(exist_ok=True)


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    result = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        result["ROC-AUC"] = roc_auc_score(y_test, y_proba)
    else:
        result["ROC-AUC"] = np.nan

    return result


def get_learning_curve(model, X, y, scoring):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )

    return {
        "train_sizes": train_sizes,
        "train_scores": train_scores.mean(axis=1),
        "validation_scores": val_scores.mean(axis=1),
    }


# CLASIFICARE
cls_df = pd.read_csv(DATA_DIR / "dataset_traffic_accident_prediction.csv")

cls_df = cls_df.dropna()
cls_df = cls_df.drop("Accident_Severity", axis=1)
cls_df["Accident"] = cls_df["Accident"].astype(int)

cls_encoded = pd.get_dummies(cls_df, drop_first=True)

X_cls = cls_encoded.drop("Accident", axis=1)
y_cls = cls_encoded["Accident"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.25,
    random_state=42,
)

classification_models = {
    "Logistic Regression": LogisticRegression(
        C=0.01,
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
    ),
    "Naive Bayes": GaussianNB(),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(C=1, kernel="rbf", class_weight="balanced", probability=True)),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
    ),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5)),
    ]),
}

classification_results = []
classification_learning_curves = {}

for name, model in classification_models.items():
    print(f"C - {name}")

    model.fit(X_train_cls, y_train_cls)

    metrics = classification_metrics(model, X_test_cls, y_test_cls)
    metrics["Model"] = name
    classification_results.append(metrics)

    classification_learning_curves[name] = get_learning_curve(
        model,
        X_train_cls,
        y_train_cls,
        scoring="f1",
    )

classification_results_df = pd.DataFrame(classification_results)
classification_results_df = classification_results_df.sort_values(by="F1", ascending=False)

joblib.dump(classification_models, MODEL_DIR / "classification_models.pkl")
joblib.dump(X_cls.columns.tolist(), MODEL_DIR / "classification_columns.pkl")
joblib.dump(classification_results_df, MODEL_DIR / "classification_results.pkl")
joblib.dump(classification_learning_curves, MODEL_DIR / "classification_learning_curves.pkl")

print("Modelele de clasificare - salvate")


# REGRESIE
reg_df = pd.read_csv(DATA_DIR / "air_quality_health_impact_data.csv")

reg_model_df = reg_df.drop(["RecordID", "HealthImpactClass"], axis=1)

X_reg = reg_model_df.drop("HealthImpactScore", axis=1)
y_reg = reg_model_df["HealthImpactScore"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.25,
    random_state=42,
)

regression_models = {
    "CatBoost Regressor": CatBoostRegressor(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=0,
    ),
    "XGBoost Regressor": XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    ),
    "Explainable Boosting Regressor": ExplainableBoostingRegressor(
        random_state=42,
        max_bins=64,
        learning_rate=0.05,
        max_rounds=100,
        interactions=0
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
    ),
    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        random_state=42,
    ),
}

regression_results = []
regression_learning_curves = {}

for name, model in regression_models.items():
    print(f"R - {name}")

    model.fit(X_train_reg, y_train_reg)

    y_pred = model.predict(X_test_reg)
    metrics = regression_metrics(y_test_reg, y_pred)
    metrics["Model"] = name
    regression_results.append(metrics)

    regression_learning_curves[name] = get_learning_curve(
        model,
        X_train_reg,
        y_train_reg,
        scoring="r2",
    )

regression_results_df = pd.DataFrame(regression_results)
regression_results_df = regression_results_df.sort_values(by="R2", ascending=False)

joblib.dump(regression_models, MODEL_DIR / "regression_models.pkl")
joblib.dump(X_reg.columns.tolist(), MODEL_DIR / "regression_columns.pkl")
joblib.dump(regression_results_df, MODEL_DIR / "regression_results.pkl")
joblib.dump(regression_learning_curves, MODEL_DIR / "regression_learning_curves.pkl")