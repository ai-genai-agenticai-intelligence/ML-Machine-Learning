from pathlib import Path
import pickle

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Salary Estimator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(
    r"D:\ABHI-VSCODE\2.MACHINE LEARNING\linear_regression_model.pkl"
).with_name("linear_regression_model.pkl")
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    .stApp {
        background: #f4f7fb;
        color: #17243a;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.9);
    }
    [data-testid="stSidebar"] {
        background: #101c31;
    }
    [data-testid="stSidebar"] * {
        color: #e8eef8;
    }
    .hero {
        padding: 2.2rem 2.5rem;
        border-radius: 22px;
        background: linear-gradient(115deg, #14233c 0%, #203e67 65%, #2c5b88 100%);
        color: #ffffff;
        margin: 0.5rem 0 1.6rem;
    }
    .hero-kicker {
        color: #a9c5e9;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
    .hero h1 {
        color: #ffffff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
    }
    .hero p {
        color: #d2deee;
        font-size: 1.02rem;
        margin: 0.65rem 0 0;
        max-width: 650px;
    }
    .section-title {
        color: #17243a;
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0.2rem 0 0.35rem;
    }
    .muted {
        color: #718096;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e5ebf3;
        border-radius: 18px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 8px 28px rgba(29, 51, 82, 0.06);
    }
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #2864a5;
        border: 0;
        border-radius: 10px;
        color: #ffffff;
        font-weight: 700;
        min-height: 2.8rem;
        transition: background 150ms ease, transform 150ms ease;
        width: 100%;
    }
    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1c4f87;
        color: #ffffff;
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5ebf3;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 8px 28px rgba(29, 51, 82, 0.05);
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #17243a !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #52627a !important;
    }
    .note {
        background: #eaf1f9;
        border-left: 4px solid #5184bd;
        border-radius: 8px;
        color: #40536d;
        padding: 0.85rem 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Salary Estimator")
    st.markdown("---")
    st.markdown("### How it works")
    st.write(
        "Enter years of experience and the linear regression model will estimate "
        "a salary from the relationship learned in the training data."
    )
    st.markdown("---")
    st.caption("Model · Linear Regression")
    st.caption("Input · Years of experience")
    st.caption("Output · Estimated annual salary")

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Machine learning · Regression</div>
        <h1>Salary, made a little clearer.</h1>
        <p>Explore an estimated salary based on years of professional experience.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_column, result_column = st.columns([1, 1.15], gap="large")

with input_column:
    st.markdown(
        '<div class="section-title">Your experience</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="muted">Set an experience level to generate an estimate.</div>',
        unsafe_allow_html=True,
    )
    with st.form("salary_prediction_form"):
        years_experience = st.slider(
            "Years of experience",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.5,
            format="%.1f years",
        )
        submitted = st.form_submit_button("Estimate salary")
    st.markdown(
        '<div class="note">This is a model-based estimate, not a guaranteed salary. '
        "Predictions beyond the training data range may be less reliable.</div>",
        unsafe_allow_html=True,
    )

with result_column:
    st.markdown(
        '<div class="section-title">Your estimate</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="muted">Your result will appear here after you submit.</div>',
        unsafe_allow_html=True,
    )
    if submitted:
        experience_input = np.array([[years_experience]])
        predicted_salary = float(model.predict(experience_input)[0])
        st.metric("Estimated annual salary", f"${predicted_salary:,.2f}")
        st.caption(f"Based on {years_experience:g} years of experience.")
    else:
        st.info(
            "Choose an experience level and select **Estimate salary** to see a result."
        )

st.markdown("---")
st.caption(
    "This demo uses a simple linear regression model trained on salary and "
    "experience data. Actual compensation can vary with role, location, and other factors."
)
