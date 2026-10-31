
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import prepare_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "return_risk_pipeline.joblib"
DATA_PATH = ROOT / "data" / "product_returns.csv"

st.set_page_config(
    page_title="Product Return Risk Prediction",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Product Return Risk Prediction")
st.caption("Machine Learning Based E-Commerce Return Analysis")

# Load dataset
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

# Load model
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_model()

# Three dashboard tabs
tab1, tab2, tab3 = st.tabs([
    "🔮 Prediction",
    "📊 Graphical Analysis",
    "🤖 Model Performance"
])

# ==========================================
# TAB 1: PREDICTION
# ==========================================

with tab1:

    st.header("Predict Product Return Risk")

    if model is None:
        st.error("Model not found. Run python src/train.py first.")
    else:

        with st.form("prediction_form"):

            c1, c2, c3 = st.columns(3)

            with c1:
                category = st.selectbox(
                    "Product Category",
                    ["Beauty", "Clothing", "Electronics", "Home", "Sports"]
                )

                price = st.number_input(
                    "Product Price", 10.0, 500.0, 250.0
                )

                discount = st.slider("Discount (%)", 0, 60, 20)

                order_hour = st.slider("Order Hour", 0, 23, 14)

            with c2:
                shipping_days = st.slider("Shipping Days", 1, 14, 4)

                payment = st.selectbox(
                    "Payment Method",
                    ["COD", "Credit Card", "Debit Card", "UPI"]
                )

                shipping_type = st.selectbox(
                    "Shipping Type",
                    ["Standard", "Express", "Same Day"]
                )

                reviews = st.slider(
                    "Customer Reviews", 0, 100, 20
                )

            with c3:
                rating = st.slider(
                    "Customer Rating", 1.0, 5.0, 4.0, 0.1
                )

                previous_returns = st.slider(
                    "Previous Returns", 0, 9, 1
                )

                account_age = st.number_input(
                    "Account Age (Days)", 1, 2500, 365
                )

            submitted = st.form_submit_button(
                "Predict Return Risk",
                type="primary"
            )

        if submitted:

            order = {
                "product_category": category,
                "product_price": price,
                "discount_percent": discount,
                "order_hour": order_hour,
                "shipping_days": shipping_days,
                "payment_method": payment,
                "shipping_type": shipping_type,
                "customer_reviews": reviews,
                "customer_rating": rating,
                "previous_returns": previous_returns,
                "account_age_days": account_age
            }

            X = prepare_features(pd.DataFrame([order]))

            probability = float(
                model.predict_proba(X)[0, 1]
            )

            risk = (
                "High" if probability >= 0.65
                else "Medium" if probability >= 0.35
                else "Low"
            )

            st.divider()
            st.subheader("Prediction Result")

            a, b = st.columns(2)

            a.metric(
                "Return Probability",
                f"{probability:.1%}"
            )

            b.metric("Risk Category", risk)

            st.progress(probability)

            if risk == "High":
                st.error("High return risk detected!")
            elif risk == "Medium":
                st.warning("Moderate return risk detected.")
            else:
                st.success("Low return risk detected.")


# ==========================================
# TAB 2: GRAPHICAL DATA ANALYSIS
# ==========================================

with tab2:

    st.header("Exploratory Data Analysis")
    st.write("Visual analysis of e-commerce product returns.")

    if df is None:
        st.error(
            "Dataset not found. Run python src/data_generation.py"
        )

    else:

        sns.set_theme(style="whitegrid")

        # KPI metrics
        total_orders = len(df)
        returned_orders = int(df["returned"].sum())
        return_rate = df["returned"].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Orders", f"{total_orders:,}")
        c2.metric("Returned Orders", f"{returned_orders:,}")
        c3.metric("Return Rate", f"{return_rate:.1%}")

        st.divider()

        # 1. Category analysis
        st.subheader("1. Return Rate by Category")

        fig, ax = plt.subplots(figsize=(9, 4))

        category_rate = (
            df.groupby("product_category")["returned"].mean()
        )

        sns.barplot(
            x=category_rate.index,
            y=category_rate.values,
            ax=ax
        )

        ax.set_xlabel("Product Category")
        ax.set_ylabel("Return Rate")
        ax.set_ylim(0, 1)

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # 2. Payment method analysis
        st.subheader("2. Return Rate by Payment Method")

        fig, ax = plt.subplots(figsize=(9, 4))

        payment_rate = (
            df.groupby("payment_method")["returned"].mean()
        )

        sns.barplot(
            x=payment_rate.index,
            y=payment_rate.values,
            ax=ax
        )

        ax.set_xlabel("Payment Method")
        ax.set_ylabel("Return Rate")
        ax.set_ylim(0, 1)

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # 3. Product price distribution
        st.subheader("3. Product Price Distribution")

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.boxplot(
            data=df,
            x="returned",
            y="product_price",
            ax=ax
        )

        ax.set_xlabel("Returned (0 = No, 1 = Yes)")
        ax.set_ylabel("Product Price")

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # 4. Previous returns analysis
        st.subheader("4. Previous Returns vs Return Rate")

        fig, ax = plt.subplots(figsize=(9, 4))

        previous_rate = (
            df.groupby("previous_returns")["returned"].mean()
        )

        sns.lineplot(
            x=previous_rate.index,
            y=previous_rate.values,
            marker="o",
            ax=ax
        )

        ax.set_xlabel("Previous Returns")
        ax.set_ylabel("Return Rate")
        ax.set_ylim(0, 1)

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # 5. Shipping analysis
        st.subheader("5. Shipping Days vs Return Rate")

        fig, ax = plt.subplots(figsize=(9, 4))

        shipping_rate = (
            df.groupby("shipping_days")["returned"].mean()
        )

        sns.lineplot(
            x=shipping_rate.index,
            y=shipping_rate.values,
            marker="o",
            ax=ax
        )

        ax.set_xlabel("Shipping Days")
        ax.set_ylabel("Return Rate")
        ax.set_ylim(0, 1)

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # 6. Correlation heatmap
        st.subheader("6. Correlation Heatmap")

        numeric_df = df.select_dtypes(
            include="number"
        ).drop(
            columns=["customer_id"],
            errors="ignore"
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        ax.set_title("Feature Correlation Analysis")

        st.pyplot(fig)
        plt.close(fig)


# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================

with tab3:

    st.header("Model Performance Analysis")

    metrics_path = ROOT / "reports" / "model_comparison.csv"

    if metrics_path.exists():

        metrics = pd.read_csv(metrics_path)

        st.dataframe(
            metrics,
            width="stretch"
        )

        if "roc_auc" in metrics.columns:

            st.subheader("Model ROC-AUC Comparison")

            fig, ax = plt.subplots(figsize=(9, 4))

            sns.barplot(
                data=metrics,
                x="model",
                y="roc_auc",
                ax=ax
            )

            ax.set_ylim(0, 1)
            ax.set_ylabel("ROC-AUC")

            plt.xticks(rotation=20)

            st.pyplot(fig)
            plt.close(fig)

    else:

        st.warning(
            "Train models first using python src/train.py"
        )
