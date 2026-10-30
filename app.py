"""MachineGuard AI — an educational predictive-maintenance dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import METRICS_PATH, MODEL_PATH, FEATURE_COLUMNS, DEFAULT_ALERT_THRESHOLD
from src.data_utils import prepare_features, canonicalise_columns


st.set_page_config(
    page_title="MachineGuard AI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
      .hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(128,128,128,.25);
             border-radius: 14px; margin-bottom: 1rem;}
      .muted {opacity: .76; font-size: .92rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.20);
          padding: .8rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def score_to_band(score: float) -> str:
    if score >= 0.70:
        return "Critical"
    if score >= 0.35:
        return "High"
    if score >= 0.15:
        return "Moderate"
    return "Low"


def maintenance_checklist(row: pd.Series) -> list[str]:
    """Illustrative screening checks based on the ranges in this dataset."""
    checks = []
    if pd.notna(row["Tool wear [min]"]) and row["Tool wear [min]"] >= 200:
        checks.append("Tool wear is high in this dataset range: inspect the tool and compare with the manufacturer's service limit.")
    if pd.notna(row["Torque [Nm]"]) and row["Torque [Nm]"] >= 60:
        checks.append("Torque is on the high side of this dataset range: review load, friction, and lubrication records.")
    if (
        pd.notna(row["Process temperature [K]"])
        and pd.notna(row["Air temperature [K]"])
        and row["Process temperature [K]"] - row["Air temperature [K]"] >= 12
    ):
        checks.append("Temperature gap is relatively high: inspect cooling and temperature-sensor readings.")
    if pd.notna(row["Rotational speed [rpm]"]) and (
        row["Rotational speed [rpm]"] < 1100 or row["Rotational speed [rpm]"] > 2200
    ):
        checks.append("Rotational speed is near the edge of the demonstration range: verify the operating specification and motor readings.")
    if not checks:
        checks.append("No simple range-based check was triggered. Continue normal inspections and follow the manufacturer's maintenance schedule.")
    return checks


def get_model_or_stop():
    if not MODEL_PATH.exists():
        st.error("The trained model was not found.")
        st.markdown(
            "Open the VS Code terminal in the project folder and run:\n\n"
            "```powershell\n"
            "python scripts/download_data.py\n"
            "python -m src.train_model\n"
            "```\n\n"
            "For an offline demo, run `python -m src.train_model --data-path data/demo_machine_data.csv`."
        )
        st.stop()
    return load_model(str(MODEL_PATH))


st.sidebar.title("⚙️ MachineGuard AI")
st.sidebar.caption("Machine failure risk screening")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Single-machine check", "Batch CSV check", "Model performance", "About & limitations"],
)

model = get_model_or_stop()
metrics = load_metrics()
threshold = st.sidebar.slider(
    "Alert threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(metrics.get("default_threshold", DEFAULT_ALERT_THRESHOLD)),
    step=0.05,
    help="Lower this threshold to flag more machines. That may increase false alarms.",
)
st.sidebar.caption("The threshold is a decision setting, not a universal safety limit.")

if page == "Overview":
    st.markdown(
        "<div class='hero'><h1>MachineGuard AI</h1>"
        "<p class='muted'>An end-to-end student project for machine-condition screening and maintenance prioritisation.</p></div>",
        unsafe_allow_html=True,
    )
    st.write(
        "Unexpected machine stoppages can interrupt production, increase repair costs, "
        "and create avoidable waste. This project uses machine sensor readings to "
        "estimate a failure score and help maintenance teams decide which records "
        "deserve an earlier inspection."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", f"{metrics.get('dataset_rows', 0):,}" if metrics else "—")
    c2.metric("Failure rate in data", f"{metrics.get('failure_rate', 0):.1%}" if metrics else "—")
    c3.metric("Holdout recall", f"{metrics.get('recall', 0):.1%}" if metrics else "—")
    c4.metric("Average precision", f"{metrics.get('average_precision', 0):.3f}" if metrics else "—")

    st.subheader("How it works")
    st.markdown(
        "1. Collect six operating fields: product type, air temperature, process temperature, rotational speed, torque, and tool wear.\n"
        "2. Clean and preprocess the readings using a saved scikit-learn pipeline.\n"
        "3. Estimate a model failure score with a Random Forest classifier.\n"
        "4. Compare the score with your chosen alert threshold.\n"
        "5. Review a simple maintenance checklist and decide whether a human inspection is needed."
    )
    st.info(
        "This is a portfolio prototype, not a certified industrial safety system. "
        "It does not predict the time until failure and must not replace trained engineers, "
        "manufacturer limits, alarms, or scheduled maintenance."
    )

elif page == "Single-machine check":
    st.title("Single-machine risk assessment")
    st.caption("Enter one set of sensor readings. The values below are starting examples; use your machine's validated sensor data.")
    with st.form("single_machine_form"):
        left, right = st.columns(2)
        with left:
            machine_type = st.selectbox("Product type", ["L", "M", "H"], index=1)
            air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=350.0, value=300.0, step=0.1)
            process_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=400.0, value=310.0, step=0.1)
        with right:
            rpm = st.number_input("Rotational speed [rpm]", min_value=0, max_value=10000, value=1500, step=10)
            torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=200.0, value=40.0, step=0.5)
            tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=1000, value=100, step=1)
        submitted = st.form_submit_button("Assess machine", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }])
        score = float(model.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
        band = score_to_band(score)
        a, b, c = st.columns(3)
        a.metric("Model failure score", f"{score:.1%}")
        b.metric("Risk band", band)
        c.metric("Alert at selected threshold", "YES" if score >= threshold else "NO")
        if score >= threshold:
            st.warning("This reading crossed the selected alert threshold. Arrange a qualified inspection according to your site's procedures.")
        else:
            st.success("This reading is below the selected alert threshold. Continue normal monitoring and scheduled inspections.")
        st.subheader("Suggested inspection checklist")
        for item in maintenance_checklist(row.iloc[0]):
            st.write(f"• {item}")
        st.caption("Checklist items are simple dataset-range heuristics, not fault diagnoses or universal operating limits.")

elif page == "Batch CSV check":
    st.title("Batch machine assessment")
    st.write("Upload a CSV containing the six sensor columns. The target column is optional for scoring.")
    st.code(
        "Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]\n"
        "M,300.1,310.2,1500,40.5,100",
        language="csv",
    )
    uploaded = st.file_uploader("Upload sensor readings (.csv)", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            canonical = canonicalise_columns(raw)
            features = prepare_features(canonical)
            scores = model.predict_proba(features)[:, 1]
            result = raw.copy()
            result["MachineGuard failure score"] = scores.round(5)
            result["Risk band"] = [score_to_band(float(value)) for value in scores]
            result["Alert"] = ["REVIEW" if value >= threshold else "MONITOR" for value in scores]
            st.success(f"Scored {len(result):,} machine records.")
            summary = result["Alert"].value_counts().rename_axis("Decision").reset_index(name="Records")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.dataframe(result.head(100), use_container_width=True, hide_index=True)
            st.download_button(
                "Download scored CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="machineguard_scored_readings.csv",
                mime="text/csv",
                type="primary",
            )
            if "Machine failure" in canonical.columns:
                actual = pd.to_numeric(canonical["Machine failure"], errors="coerce")
                valid = actual.isin([0, 1])
                if valid.sum() > 0:
                    st.caption("Your upload includes labels. These are retrospective comparison metrics for this uploaded file, not a fresh independent test.")
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    actual_valid = actual.loc[valid].astype(int)
                    predicted_valid = (scores[valid.to_numpy()] >= threshold).astype(int)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precision", f"{precision_score(actual_valid, predicted_valid, zero_division=0):.1%}")
                    m2.metric("Recall", f"{recall_score(actual_valid, predicted_valid, zero_division=0):.1%}")
                    m3.metric("F1 score", f"{f1_score(actual_valid, predicted_valid, zero_division=0):.3f}")
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

elif page == "Model performance":
    st.title("Model performance")
    if not metrics:
        st.warning("Metrics report is missing. Retrain the model to create reports/metrics.json.")
    else:
        st.caption(f"Holdout evaluation • alert threshold = {metrics.get('default_threshold', DEFAULT_ALERT_THRESHOLD):.2f} • model = {metrics.get('model_name', 'Unknown')}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics['precision']:.1%}")
        m2.metric("Recall", f"{metrics['recall']:.1%}")
        m3.metric("F1", f"{metrics['f1']:.3f}")
        m4.metric("Average precision", f"{metrics['average_precision']:.3f}")

        st.subheader("Confusion matrix")
        st.caption("Rows = actual class; columns = predicted class. Class 1 means machine failure.")
        matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        st.dataframe(
            pd.DataFrame(matrix, index=["Actual: no failure", "Actual: failure"], columns=["Predicted: no failure", "Predicted: failure"]),
            use_container_width=True,
        )

        st.subheader("Permutation feature importance")
        st.caption("Importance is measured by the decrease in average precision when a feature is shuffled on the holdout set. Negative values can occur.")
        importance = pd.DataFrame(metrics.get("feature_importance", []))
        if not importance.empty:
            importance = importance.sort_values("importance", ascending=True)
            st.bar_chart(importance.set_index("feature"))
        st.subheader("Other evaluation metrics")
        st.write({
            "Accuracy": round(metrics["accuracy"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "Test rows": metrics["test_rows"],
            "Test failure rate": f"{metrics['test_failure_rate']:.2%}",
        })
        st.warning("A high accuracy score can be misleading when failures are rare. Review recall, precision, average precision, and the actual confusion matrix.")

else:
    st.title("About this project & responsible use")
    st.subheader("Industrial problem")
    st.write("Unplanned equipment downtime can disrupt production and increase maintenance cost. MachineGuard demonstrates how sensor readings can be turned into a prioritised inspection queue.")
    st.subheader("Dataset transparency")
    st.write("The main tutorial uses the UCI AI4I 2020 Predictive Maintenance dataset. UCI describes it as a synthetic dataset that reflects industrial predictive-maintenance data; it is not a sample of real factory operations.")
    st.subheader("Important limitations")
    st.markdown(
        "- The model predicts the dataset's recorded `Machine failure` label from the same observation's sensor readings. It does **not** forecast a future time window or remaining useful life.\n"
        "- The score from `predict_proba` may not be a calibrated real-world probability.\n"
        "- A false negative may be costly; a false positive may create unnecessary inspection work. The alert threshold must be selected with plant engineers using real operating costs.\n"
        "- The demo data generator is simulated and exists only for offline practice.\n"
        "- Before a real deployment, validate on time-separated data from the actual machines, check sensor quality and drift, calibrate scores, and complete safety/engineering review."
    )
    st.subheader("Recommended next steps")
    st.write("Add time-aware validation, maintenance-cost analysis, sensor-drift monitoring, model versioning, audit logs, and an authenticated API after obtaining permissioned plant data.")

st.sidebar.divider()
st.sidebar.caption("Student portfolio prototype • Human review required")
