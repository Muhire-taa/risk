"""
Streamlit app: Real-time Poverty Status Prediction using saved ANN weights.
Pure numpy inference — no TensorFlow or ONNX needed.
Target: poor_16 — 1 = Poor, 0 = Non-Poor
"""

import streamlit as st
import numpy as np
import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "poverty_model_artifacts")
MODEL_FILE = os.path.join(MODEL_DIR, "model_data.json")


@st.cache_resource
def load_model():
    with open(MODEL_FILE) as f:
        data = json.load(f)
    weights = [(np.array(w), np.array(b)) for w, b in data["weights"]]
    activations = data["activations"]
    scaler_mean = np.array(data["scaler_mean"])
    scaler_scale = np.array(data["scaler_scale"])
    features = data["features"]
    target_labels = data["target_labels"]
    return weights, activations, scaler_mean, scaler_scale, features, target_labels


def predict(X, weights, activations, scaler_mean, scaler_scale):
    x = (X - scaler_mean) / scaler_scale
    for (w, b), act in zip(weights, activations):
        x = x @ w + b
        if act == "relu":
            x = np.maximum(0, x)
        elif act == "sigmoid":
            x = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return float(x.flatten()[0])


FIELDS = [
    {
        "key": "hhedyrs",
        "name": "hhedyrs",
        "desc": "Number of school years completed - Ranges from 0 to 17",
        "type": "number",
        "min": 0, "max": 17, "default": 6,
    },
    {
        "key": "hhsex",
        "name": "hhsex",
        "desc": "Sex of the household member (Household Head) - 0 Female and 1 Male",
        "type": "select",
        "options": {"Female": 0.0, "Male": 1.0},
    },
    {
        "key": "region",
        "name": "region",
        "desc": "Region of Residence in 2016/17 - 1 Central, 2 Eastern, 3 Northern and 4 Western",
        "type": "select",
        "options": {"Central": 1.0, "Eastern": 2.0, "Northern": 3.0, "Western": 4.0},
    },
    {
        "key": "mstat",
        "name": "mstat",
        "desc": "Marital status of household head - 1 Married Monogamous, 2 Married Polygamous, 3 Divorced/Separated, 4 Widow/Widower and 5 Never Married",
        "type": "select",
        "options": {
            "Married Monogamous": 1.0,
            "Married Polygamous": 2.0,
            "Divorced / Separated": 3.0,
            "Widow / Widower": 4.0,
            "Never Married": 5.0,
        },
    },
    {
        "key": "hhage",
        "name": "hhage",
        "desc": "Age in completed years - Ranges from 11 to 110 years",
        "type": "number",
        "min": 11, "max": 110, "default": 40,
    },
    {
        "key": "hsize",
        "name": "hsize",
        "desc": "Household Size - Ranges from 1 to 23",
        "type": "number",
        "min": 1, "max": 23, "default": 4,
    },
    {
        "key": "urban",
        "name": "urban",
        "desc": "Urban/Rural Identifier - 0 Rural and 1 Urban",
        "type": "select",
        "options": {"Rural": 0.0, "Urban": 1.0},
    },
    {
        "key": "CB02",
        "name": "CB02",
        "desc": "Frequency of receiving the money from the main source of income - 1 Daily, 2 Weekly, 3 Monthly, 4 Seasonally, 5 Annually and 6 Irregularly",
        "type": "select",
        "options": {
            "Daily": 1.0,
            "Weekly": 2.0,
            "Monthly": 3.0,
            "Seasonally": 4.0,
            "Annually": 5.0,
            "Irregularly": 6.0,
        },
    },
    {
        "key": "pwall",
        "name": "pwall",
        "desc": "Major construction material for wall - 0 Others and 1 Permanent",
        "type": "select",
        "options": {"Others": 0.0, "Permanent": 1.0},
    },
    {
        "key": "pfloor",
        "name": "pfloor",
        "desc": "Major construction material for floor - 0 Others and 1 Permanent",
        "type": "select",
        "options": {"Others": 0.0, "Permanent": 1.0},
    },
    {
        "key": "room",
        "name": "room",
        "desc": "Number of rooms used for sleeping - Ranges from 1 to 12 rooms",
        "type": "number",
        "min": 1, "max": 12, "default": 2,
    },
    {
        "key": "ha03_11",
        "name": "ha03_11",
        "desc": "Any member of your household own an asset currently: Mobile phone - 1 Yes individually, 3 No, 4 Yes Jointly with household member and 5 Yes Jointly with a non-household member",
        "type": "select",
        "options": {
            "Yes, individually": 1.0,
            "No": 3.0,
            "Yes, jointly with household member": 4.0,
            "Yes, jointly with non-household member": 5.0,
        },
    },
    {
        "key": "ha03_9",
        "name": "ha03_9",
        "desc": "Any member of your household own an asset currently: Radio - 1 Yes individually, 3 No, 4 Yes Jointly with household member and 5 Yes Jointly with a non-household member",
        "type": "select",
        "options": {
            "Yes, individually": 1.0,
            "No": 3.0,
            "Yes, jointly with household member": 4.0,
            "Yes, jointly with non-household member": 5.0,
        },
    },
    {
        "key": "IT5_3",
        "name": "IT5_3",
        "desc": "Distance in KMs to the nearest Financial services - 1 (0 to <3 KMs), 2 (3 to <5 KMs), 3 (5 to <8 KMs) and 4 (8 or more KMs)",
        "type": "select",
        "options": {
            "0 to <3 KMs": 1.0,
            "3 to <5 KMs": 2.0,
            "5 to <8 KMs": 3.0,
            "8 or more KMs": 4.0,
        },
    },
    {
        "key": "HC08a",
        "name": "HC08a",
        "desc": "Time taken to and from the source of drinking water - Ranges from 0 to 360 minutes",
        "type": "number",
        "min": 0, "max": 360, "default": 20,
    },
]


def main():
    st.set_page_config(page_title="Poverty Status Predictor", page_icon="📊", layout="centered")

    st.markdown(
        """
        <style>
        .block-container { max-width: 95%; padding-left: 2rem; padding-right: 2rem; }
        .stApp { background-color: #fff !important; }
        .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp p,
        .stApp label, .stApp span, .stApp div,
        .stApp .stMarkdown, [data-testid="stCaptionContainer"] p {
            color: #1a1a1a !important;
        }
        .stApp [data-testid="stFormSubmitButton"] button {
            color: #fff !important;
        }
        .field-row {
            border-bottom: 1px solid #ebebeb;
            padding: 18px 0 12px 0;
        }
        .field-name {
            font-weight: 700;
            font-size: 0.95rem;
            color: #000 !important;
        }
        .stApp .field-required {
            color: #e00 !important;
            font-size: 0.72rem;
            font-weight: 400;
            font-style: italic;
            margin-left: 6px;
        }
        .stApp .field-type {
            font-size: 0.78rem;
            color: #aaa !important;
            margin-top: 1px;
        }
        .field-type i {
            font-style: italic;
        }
        .field-desc {
            font-size: 0.88rem;
            color: #000 !important;
            margin: 8px 0 10px 0;
            line-height: 1.5;
        }
        .result-card {
            border-radius: 12px;
            padding: 20px 24px;
            text-align: center;
            margin-bottom: 8px;
        }
        .result-card .label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .result-card .value {
            font-size: 2rem;
            font-weight: 800;
        }
        .card-poor { background: #fee2e2; border: 2px solid #ef4444; }
        .stApp .card-poor .label { color: #991b1b !important; }
        .stApp .card-poor .value { color: #dc2626 !important; }
        .card-nonpoor { background: #d1fae5; border: 2px solid #10b981; }
        .stApp .card-nonpoor .label { color: #065f46 !important; }
        .stApp .card-nonpoor .value { color: #059669 !important; }
        .card-prob { background: #dbeafe; border: 2px solid #3b82f6; }
        .stApp .card-prob .label { color: #1e3a5f !important; }
        .stApp .card-prob .value { color: #1d4ed8 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Poverty Status Predictor")
    st.caption("Deployed ML model using Streamlit")

    if not os.path.isfile(MODEL_FILE):
        st.error(
            "**Model data not found.** To fix this:\n\n"
            "1. Run all cells in `model_validation.ipynb` (including the **export** and **embed** cells).\n"
            "2. In your local terminal, run: `python3 extract_model.py`\n"
            "3. Reload this page."
        )
        st.stop()

    values = {}

    with st.form("predict"):
        st.markdown(
            '<div style="display:flex; padding:10px 0; border-bottom:2px solid #dedede; margin-bottom:4px;">'
            '<span style="width:120px; font-weight:700; font-size:0.85rem; color:#000;">Name</span>'
            '<span style="font-weight:700; font-size:0.85rem; color:#000;">Description</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        for field in FIELDS:
            st.markdown(
                f'<div class="field-row">'
                f'<span class="field-name">{field["name"]}</span>'
                f'<span class="field-required"> * required</span><br>'
                f'<span class="field-type">string<br><i>(query)</i></span>'
                f'<div class="field-desc">{field["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if field["type"] == "select":
                opts = field["options"]
                selected = st.selectbox(
                    f'{field["name"]}',
                    list(opts.keys()),
                    key=field["key"],
                    label_visibility="collapsed",
                )
                values[field["key"]] = opts[selected]
            else:
                val = st.number_input(
                    f'{field["name"]}',
                    min_value=field["min"],
                    max_value=field["max"],
                    value=field["default"],
                    key=field["key"],
                    label_visibility="collapsed",
                )
                values[field["key"]] = float(val)

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        weights, activations, scaler_mean, scaler_scale, features, target_labels = load_model()

        X = np.array([[values[f] for f in features]], dtype=np.float64)
        prob = predict(X, weights, activations, scaler_mean, scaler_scale)
        pred = 1 if prob >= 0.5 else 0
        label = target_labels[str(pred)]

        st.divider()

        card_class = "card-poor" if pred == 1 else "card-nonpoor"
        st.markdown(
            f'<div class="result-card {card_class}">'
            f'<div class="label">Predicted Status</div>'
            f'<div class="value">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
