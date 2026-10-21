"""Page 1: Diabetes Risk (classification track, final model from the notebook)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from glucotrack import data, ui
from glucotrack.pipeline import (CLASSIFICATION_RAW_COLUMNS, engineer_classification_features,
                                 preprocess_classification)

P = "risk_"
SMOKING_ORDER = ["never", "former", "current", "not current", "ever", "No Info"]

EXAMPLES = {
    "Example A: mid-range values": dict(age=45.0, gender="Female", location=None, race="Caucasian", smoking="never",
                                         hypertension=False, heart=False, bmi=26.5, hba1c=5.6, glucose=120),
    "Example B: elevated values": dict(age=62.0, gender="Male", location=None, race="Hispanic", smoking="former",
                                        hypertension=True, heart=False, bmi=33.0, hba1c=6.3, glucose=165),
}


def _defaults(stats: dict) -> dict:
    r = stats["ranges"]
    return {
        "age": float(round(r["age"]["median"])), "gender": "Female", "location": stats["categories"]["location"][0],
        "race": "Caucasian", "smoking": "never", "hypertension": False, "heart": False,
        "bmi": round(r["bmi"]["median"], 1), "hba1c": round(r["hbA1c_level"]["median"], 1),
        "glucose": int(r["blood_glucose_level"]["median"]),
    }


def _init(stats: dict) -> None:
    for k, v in _defaults(stats).items():
        st.session_state.setdefault(P + k, v)


def _load_example(name: str, stats: dict) -> None:
    ex = dict(_defaults(stats)); ex.update({k: v for k, v in EXAMPLES[name].items() if v is not None})
    for k, v in ex.items():
        st.session_state[P + k] = v
    st.session_state.pop(P + "result", None)


def _reset(stats: dict) -> None:
    for k, v in _defaults(stats).items():
        st.session_state[P + k] = v
    st.session_state.pop(P + "result", None)


def _current_inputs() -> dict:
    s = st.session_state
    return {
        "gender": s[P + "gender"], "age": float(s[P + "age"]), "location": s[P + "location"],
        "hypertension": int(s[P + "hypertension"]), "heart_disease": int(s[P + "heart"]),
        "smoking_history": s[P + "smoking"], "bmi": float(s[P + "bmi"]), "hbA1c_level": float(s[P + "hba1c"]),
        "blood_glucose_level": float(s[P + "glucose"]), "race": s[P + "race"],
    }


def _analyze() -> None:
    inputs = _current_inputs()
    try:
        b = data.bundle("classification")
        raw = pd.DataFrame([inputs])[CLASSIFICATION_RAW_COLUMNS]
        X = preprocess_classification(raw, b)
        proba = float(b["model"].predict_proba(X)[0, 1])
        derived = engineer_classification_features(raw).iloc[0]
        st.session_state[P + "result"] = {
            "inputs": inputs, "proba": proba, "pred": int(proba >= 0.5), "model": b["model_name"],
            "derived": {"BMI category": derived["bmi_category"],
                        "Age x HbA1c interaction": f"{derived['age_risk_interaction']:.2f}",
                        "Metabolic risk score (0 to 5)": str(int(derived["metabolic_risk_score"]))},
        }
        st.session_state.pop(P + "error", None)
    except Exception as exc:  # surfaced in the results panel instead of a stack trace
        st.session_state[P + "error"] = f"{type(exc).__name__}: {exc}"
        st.session_state.pop(P + "result", None)


def render() -> None:
    stats = data.jsonf("classification", "stats.json")
    b = data.bundle("classification")
    r = stats["ranges"]
    _init(stats)

    ui.page_header("Diabetes Risk",
                   "Enter a patient profile and see what the final classifier from the capstone predicts. "
                   "This page demonstrates the classification track.")

    left, right = st.columns([5, 6], gap="large")

    # ------------------------------------------------------------------ inputs
    with left:
        with ui.card("risk-form", "Patient profile", "All fields use the same columns and units as the training data."):
            ui.section("Demographics")
            c1, c2 = st.columns(2)
            c1.number_input("Age (years)", min_value=float(r["age"]["min"]), max_value=float(r["age"]["max"]),
                            step=1.0, key=P + "age", help=f"Training range {r['age']['min']} to {r['age']['max']:.0f} years.")
            c2.selectbox("Gender", stats["categories"]["gender"], key=P + "gender")
            c3, c4 = st.columns(2)
            c3.selectbox("Race / ethnicity", stats["categories"]["race"], key=P + "race", format_func=ui.label)
            c4.selectbox("State / territory", stats["categories"]["location"], key=P + "location",
                         help="Used by the model as a frequency-encoded feature (how common that location is in the training data).")
            ui.section("Clinical measurements")
            c5, c6 = st.columns(2)
            c5.number_input("BMI (kg/m²)", min_value=float(r["bmi"]["min"]), max_value=float(r["bmi"]["max"]),
                            step=0.1, key=P + "bmi", help=f"Training range {r['bmi']['min']} to {r['bmi']['max']}.")
            c6.selectbox("Smoking history", [s for s in SMOKING_ORDER if s in stats["categories"]["smoking_history"]],
                         key=P + "smoking", format_func=ui.label)
            st.slider("HbA1c (%)", min_value=float(r["hbA1c_level"]["min"]), max_value=float(r["hbA1c_level"]["max"]),
                      step=0.1, key=P + "hba1c", help="Glycated haemoglobin. Training range 3.5 to 9.0 %.")
            st.slider("Blood glucose (mg/dL)", min_value=int(r["blood_glucose_level"]["min"]),
                      max_value=int(r["blood_glucose_level"]["max"]), step=1, key=P + "glucose",
                      help="Training range 80 to 300. Units are assumed from the value range; the source does not state them.")
            ui.section("Comorbidities")
            c7, c8 = st.columns(2)
            c7.toggle("Hypertension", key=P + "hypertension")
            c8.toggle("Heart disease", key=P + "heart")

            st.button("Analyze Patient", type="primary", width="stretch", on_click=_analyze, key=P + "go")
            e1, e2 = st.columns(2)
            e1.button("Reset", width="stretch", on_click=_reset, args=(stats,), key=P + "reset")
            e2.selectbox("Load an illustrative example", ["Choose…"] + list(EXAMPLES), key=P + "example",
                         label_visibility="collapsed",
                         on_change=lambda: _load_example(st.session_state[P + "example"], stats)
                         if st.session_state[P + "example"] in EXAMPLES else None)
            ui.caption("Examples are illustrative value sets, not real patients and not clinically validated profiles.")

    # ------------------------------------------------------------------ results
    with right:
        res = st.session_state.get(P + "result")
        err = st.session_state.get(P + "error")
        if err:
            with ui.card("risk-error"):
                ui.callout("alert", "The prediction could not be computed", f"{ui.esc(err)}. Check the inputs and try again.")
        elif res is None:
            with ui.card("risk-empty", "Result"):
                ui.empty_state("No prediction yet",
                               "Fill in the patient profile and press Analyze Patient, or load an illustrative example from the selector.")
                ui.callout("info", "What the model sees",
                           f"The final model is <b>{ui.esc(b['model_name'])}</b>, selected from ten algorithms. "
                           f"On the held-out test set it reached weighted F1 "
                           f"{data.table('classification', 'metrics.csv').iloc[0]['F1 (weighted)']:.4f}; see Model Comparison.")
        else:
            _results(res, stats, b)


def _results(res: dict, stats: dict, b: dict) -> None:
    inputs, proba, pred = res["inputs"], res["proba"], res["pred"]
    stale = inputs != _current_inputs()
    rule = stats["label_rule"]
    metrics = data.table("classification", "metrics.csv")
    m = metrics[metrics["Model"] == res["model"]].iloc[0]

    with ui.card("risk-result", "Result", None):
        st.markdown(ui.badge(f"Model: {res['model']}", "teal") + " " + ui.badge("Test ROC-AUC " + f"{m['ROC-AUC']:.4f}"),
                    unsafe_allow_html=True)
        if stale:
            ui.callout("warn", "Inputs changed", "The values on the left differ from the ones analysed below. Press Analyze Patient to update.")
        ui.big_result("Estimated probability of the diabetic class", f"{proba:.1%}")
        ui.prob_bar(proba)
        cls = "diabetic" if pred else "non-diabetic"
        st.markdown(
            f'<p class="gt-body" style="margin-top:12px"><b>Predicted class: {cls}.</b> '
            f'The model assigns this record to the {cls} class because its estimated probability is '
            f'{"at or above" if pred else "below"} the 50% decision threshold. '
            f'This is a model output on the entered values, not a diagnosis.</p>', unsafe_allow_html=True)

        ui.callout("neutral", "How to read the probability",
                   "It is the classifier's score for the diabetic class. Tree ensembles are not guaranteed to be "
                   "calibrated, so treat it as a ranking score rather than a clinical risk percentage.")

        if inputs["blood_glucose_level"] > rule["max_glucose_nondiabetic"] or inputs["hbA1c_level"] > rule["max_hba1c_nondiabetic"]:
            ui.callout("alert", "This input sits inside the dataset's labelling rule",
                       f"In the training data every record with glucose above {rule['max_glucose_nondiabetic']:.0f} "
                       f"({rule['n_glucose_above_200']:,} records) or HbA1c above {rule['max_hba1c_nondiabetic']:.1f} "
                       f"({rule['n_hba1c_above_6_6']:,} records) is labelled diabetic. A high probability here mostly "
                       f"reflects that rule rather than a learned clinical relationship.")

        t1, t2 = st.tabs(["Inputs used", "Derived features"])
        with t1:
            ui.kv_table([
                ("Age", f"{inputs['age']:.1f} years"), ("Gender", inputs["gender"]),
                ("Race / ethnicity", ui.label(inputs["race"])), ("State / territory", inputs["location"]),
                ("BMI", f"{inputs['bmi']:.1f} kg/m²"), ("Smoking history", ui.label(inputs["smoking_history"])),
                ("HbA1c", f"{inputs['hbA1c_level']:.1f} %"), ("Blood glucose", f"{inputs['blood_glucose_level']:.0f} mg/dL"),
                ("Hypertension", "Yes" if inputs["hypertension"] else "No"),
                ("Heart disease", "Yes" if inputs["heart_disease"] else "No"),
            ])
        with t2:
            ui.kv_table(list(res["derived"].items()))
            ui.caption("Computed from the inputs by the same code as the notebook (section 3.5) before the model is applied.")

        out = pd.DataFrame([{**inputs, "predicted_class": cls, "probability_diabetic_class": round(proba, 6),
                             "model": res["model"], "decision_threshold": 0.5}])
        ui.csv_download("Download result (CSV)", out, "glucotrack_risk_prediction.csv", P + "dl")

    with ui.card("risk-leak", "Important limitation: target leakage"):
        ui.callout("warn", "The model may be learning the labelling rule",
                   f"In this dataset the diabetes label appears to be assigned by thresholds: the highest glucose among "
                   f"non-diabetic records is exactly {rule['max_glucose_nondiabetic']:.0f}, and the highest HbA1c is "
                   f"{rule['max_hba1c_nondiabetic']:.1f}. Because glucose and HbA1c are also inputs, high test scores partly "
                   f"measure how well the model recovers that rule. The model is <b>not clinically validated</b> and its "
                   f"performance should not be expected to transfer to real patients. "
                   f"See the About page for details.")
