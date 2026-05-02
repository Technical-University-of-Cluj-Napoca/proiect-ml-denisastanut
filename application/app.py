from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score


st.set_page_config(page_title="Proiect ML", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "CSV_data"
MODEL_DIR = BASE_DIR / "app_models"


@st.cache_data
def load_data():
    cls_df = pd.read_csv(DATA_DIR / "dataset_traffic_accident_prediction.csv")
    reg_df = pd.read_csv(DATA_DIR / "air_quality_health_impact_data.csv")
    return cls_df, reg_df


@st.cache_resource
def load_artifacts():
    artifacts = {
        "cls_models": joblib.load(MODEL_DIR / "classification_models.pkl"),
        "cls_cols": joblib.load(MODEL_DIR / "classification_columns.pkl"),
        "cls_results": joblib.load(MODEL_DIR / "classification_results.pkl"),
        "cls_curves": joblib.load(MODEL_DIR / "classification_learning_curves.pkl"),
        "reg_models": joblib.load(MODEL_DIR / "regression_models.pkl"),
        "reg_cols": joblib.load(MODEL_DIR / "regression_columns.pkl"),
        "reg_results": joblib.load(MODEL_DIR / "regression_results.pkl"),
        "reg_curves": joblib.load(MODEL_DIR / "regression_learning_curves.pkl"),
    }
    return artifacts


cls_df, reg_df = load_data()
art = load_artifacts()


def metric_table(df: pd.DataFrame):
    st.dataframe(df.round(4), use_container_width=True, hide_index=True)


def plot_learning_curve(curve: dict, title: str, ylabel: str):
    if len(curve["train_sizes"]) == 0:
        st.info("Curba de invatare nu a fost salvata pentru acest model")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["train_sizes"], curve["train_scores"], marker="o", label="Train")
    ax.plot(curve["train_sizes"], curve["validation_scores"], marker="o", label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Training size")
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)


def model_params(model):
    if hasattr(model, "get_params"):
        params = model.get_params()
        params_df = pd.DataFrame(list(params.items()), columns=["Parametru", "Valoare"])
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    else:
        st.write("Nu exista hiperparametri disponibili pentru acest model")


def prepare_cls_data(df):
    df = df.dropna()
    df = df.drop("Accident_Severity", axis=1)
    df["Accident"] = df["Accident"].astype(int)

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("Accident", axis=1)
    y = df_encoded["Accident"]

    return X, y


def prepare_reg_data(df):
    df = df.drop(["RecordID", "HealthImpactClass"], axis=1)
    X = df.drop("HealthImpactScore", axis=1)
    y = df["HealthImpactScore"]
    return X, y


def make_cls_input(values: dict):
    input_df = pd.DataFrame(columns=art["cls_cols"])
    input_df.loc[0] = 0

    numeric_cols = [
        "Traffic_Density",
        "Speed_Limit",
        "Number_of_Vehicles",
        "Driver_Alcohol",
        "Driver_Age",
        "Driver_Experience",
    ]

    for col in numeric_cols:
        if col in input_df.columns:
            input_df.loc[0, col] = values[col]

    categorical_cols = [
        "Weather",
        "Road_Type",
        "Time_of_Day",
        "Road_Condition",
        "Vehicle_Type",
        "Road_Light_Condition",
    ]

    for col in categorical_cols:
        dummy_col = f"{col}_{values[col]}"
        if dummy_col in input_df.columns:
            input_df.loc[0, dummy_col] = 1

    return input_df


def classification_page():
    st.title("Clasificare - predictia accidentelor rutiere")

    st.write(
        """
        Aceasta pagina prezinta problema de clasificare. Scopul este sa prezicem daca are loc sau nu un accident rutier,
        folosind informatii despre vreme, drum, trafic si sofer.
        """
    )

    model_names = art["cls_results"]["Model"].tolist()
    selected_model = st.selectbox("Alege modelul de clasificare", model_names)
    model = art["cls_models"][selected_model]

    tabs = st.tabs(["Date si EDA", "Modele si evaluare", "Predictie si SHAP"])

    with tabs[0]:
        st.subheader("Dataset")
        st.dataframe(cls_df.head(), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Randuri", cls_df.shape[0])
        c2.metric("Coloane", cls_df.shape[1])
        c3.metric("Valori lipsa", int(cls_df.isnull().sum().sum()))

        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(x="Accident", data=cls_df, ax=ax)
            ax.set_title("Distributia target-ului")
            st.pyplot(fig)

        with right:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(cls_df["Speed_Limit"].dropna(), bins=30, kde=True, ax=ax)
            ax.set_title("Distributia limitei de viteza")
            st.pyplot(fig)

    with tabs[1]:
        st.subheader("Tabel comparativ modele")
        metric_table(art["cls_results"])

        st.subheader(f"Metrici pentru modelul selectat: {selected_model}")
        selected_row = art["cls_results"][art["cls_results"]["Model"] == selected_model]
        metric_table(selected_row)

        X, y = prepare_cls_data(cls_df)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Matricea de confuzie")
        ax.set_xlabel("Predictie")
        ax.set_ylabel("Valoare reala")
        st.pyplot(fig)

        st.subheader("Curba de invatare")
        plot_learning_curve(
            art["cls_curves"][selected_model],
            f"Learning Curve - {selected_model}",
            "F1-score",
        )

        st.subheader("Hiperparametri model")
        model_params(model)

    with tabs[2]:
        st.subheader("Predictie interactiva")

        clean_df = cls_df.dropna()

        with st.form("cls_form"):
            left, right = st.columns(2)

            with left:
                weather = st.selectbox("Weather", sorted(clean_df["Weather"].unique()))
                road_type = st.selectbox("Road Type", sorted(clean_df["Road_Type"].unique()))
                time_of_day = st.selectbox("Time of Day", sorted(clean_df["Time_of_Day"].unique()))
                road_condition = st.selectbox("Road Condition", sorted(clean_df["Road_Condition"].unique()))
                vehicle_type = st.selectbox("Vehicle Type", sorted(clean_df["Vehicle_Type"].unique()))
                road_light = st.selectbox("Road Light Condition", sorted(clean_df["Road_Light_Condition"].unique()))

            with right:
                traffic_density = st.slider("Traffic Density", 0.0, 2.0, 1.0)
                speed_limit = st.slider("Speed Limit", 30.0, 220.0, 60.0)
                vehicles = st.slider("Number of Vehicles", 1.0, 15.0, 3.0)
                alcohol = st.selectbox("Driver Alcohol", [0.0, 1.0])
                age = st.slider("Driver Age", 18.0, 70.0, 35.0)
                experience = st.slider("Driver Experience", 0.0, 70.0, 10.0)

            submitted = st.form_submit_button("Genereaza predictia")

        if submitted:
            values = {
                "Weather": weather,
                "Road_Type": road_type,
                "Time_of_Day": time_of_day,
                "Traffic_Density": traffic_density,
                "Speed_Limit": speed_limit,
                "Number_of_Vehicles": vehicles,
                "Driver_Alcohol": alcohol,
                "Road_Condition": road_condition,
                "Vehicle_Type": vehicle_type,
                "Driver_Age": age,
                "Driver_Experience": experience,
                "Road_Light_Condition": road_light,
            }

            input_df = make_cls_input(values)
            pred = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]
                st.metric("Probabilitate accident", round(prob, 3))

            if pred == 1:
                st.error("Predictie: risc de accident")
            else:
                st.success("Predictie: nu are loc accident")


def regression_page():
    st.title("Regresie - impactul calitatii aerului asupra sanatatii")

    st.write(
        """
        Aceasta pagina prezinta problema de regresie. Scopul este sa prezicem valoarea `HealthImpactScore`,
        folosind date despre poluanti, conditii meteo si cazuri medicale.
        """
    )

    model_names = art["reg_results"]["Model"].tolist()
    selected_model = st.selectbox("Alege modelul de regresie", model_names)
    model = art["reg_models"][selected_model]

    tabs = st.tabs(["Date si EDA", "Modele si evaluare", "Predictie si SHAP"])

    with tabs[0]:
        st.subheader("Dataset")
        st.dataframe(reg_df.head(), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Randuri", reg_df.shape[0])
        c2.metric("Coloane", reg_df.shape[1])
        c3.metric("Valori lipsa", int(reg_df.isnull().sum().sum()))

        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(reg_df["HealthImpactScore"], bins=30, kde=True, ax=ax)
            ax.set_title("Distributia HealthImpactScore")
            st.pyplot(fig)

        with right:
            corr = reg_df.corr()["HealthImpactScore"].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            corr.drop("HealthImpactScore").plot(kind="bar", ax=ax)
            ax.set_title("Corelatii cu target-ul")
            st.pyplot(fig)

    with tabs[1]:
        st.subheader("Tabel comparativ modele")
        metric_table(art["reg_results"])

        st.subheader(f"Metrici pentru modelul selectat: {selected_model}")
        selected_row = art["reg_results"][art["reg_results"]["Model"] == selected_model]
        metric_table(selected_row)

        X, y = prepare_reg_data(reg_df)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE", round(mse, 3))
        c2.metric("MAE", round(mae, 3))
        c3.metric("RMSE", round(rmse, 3))
        c4.metric("R2", round(r2, 3))

        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
            ax.set_title("Valori reale vs prezise")
            ax.set_xlabel("Valori reale")
            ax.set_ylabel("Valori prezise")
            st.pyplot(fig)

        with right:
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(residuals, bins=30, kde=True, ax=ax)
            ax.set_title("Distributia erorilor")
            st.pyplot(fig)

        st.subheader("Curba de invatare")
        plot_learning_curve(
            art["reg_curves"][selected_model],
            f"Learning Curve - {selected_model}",
            "R2-score",
        )

        st.subheader("Hiperparametri model")
        model_params(model)

    with tabs[2]:
        st.subheader("Predictie interactiva")

        values = {}

        with st.form("reg_form"):
            left, right = st.columns(2)

            for i, col in enumerate(art["reg_cols"]):
                min_val = float(reg_df[col].min())
                max_val = float(reg_df[col].max())
                mean_val = float(reg_df[col].mean())

                target_col = left if i % 2 == 0 else right

                with target_col:
                    values[col] = st.slider(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                    )

            submitted = st.form_submit_button("Genereaza predictia")

        if submitted:
            input_df = pd.DataFrame([values])
            input_df = input_df[art["reg_cols"]]

            prediction = model.predict(input_df)[0]

            st.success(f"HealthImpactScore prezis: {prediction:.2f}")

            st.subheader("SHAP pentru predictia generata")

            try:
                import shap

                explainer = shap.Explainer(model, input_df)
                shap_values = explainer(input_df)

                fig = plt.figure()
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)


pages = {
    "Clasificare": classification_page,
    "Regresie": regression_page,
}

st.sidebar.title("Meniu")
selected_page = st.sidebar.radio("Alege pagina:", list(pages.keys()))

pages[selected_page]()