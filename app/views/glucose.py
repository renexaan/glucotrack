"""Page 2: Glucose Estimate (regression track, best model by test-set R²)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from glucotrack import charts, data, ui
from glucotrack.pipeline import REGRESSION_RAW_COLUMNS, preprocess_regression

P = "glu_"
SMOKING_ORDER = ["never", "former", "current", "not current", "ever", "No Info"]
EXAMPLES = {
    "Example A: lower-range values": dict(age=34.0, gender="Female", smoking="never", hypertension=False, heart=False,
                                          bmi=23.5, hba1c=5.2, diabetes=False),
    "Example B: higher-range values": dict(age=61.0, gender="Male", smoking="former", hypertension=True, heart=False,
                                           bmi=32.0, hba1c=6.5, diabetes=True),
}


def _defaults(stats: dict) -> dict:
    r = stats["ranges"]
    return {"age": float(round(r["age"]["median"])), "gender": "Female", "smoking": "never", "hypertension": False,
            "heart": False, "bmi": round(r["bmi"]["median"], 1), "hba1c": round(r["HbA1c_level"]["median"], 1),
            "diabetes": False}


def _init(stats: dict) -> None:
    for k, v in _defaults(stats).items():
        st.session_state.setdefault(P + k, v)


def _load_example(name: str, stats: dict) -> None:
    ex = _defaults(stats); ex.update(EXAMPLES[name])
    for k, v in ex.items():
        st.session_state[P + k] = v
    st.session_state.pop(P + "result", None)


def _reset(stats: dict) -> None:
    for k, v in _defaults(stats).items():
        st.session_state[P + k] = v
    st.session_state.pop(P + "result", None)
    st.session_state[P + "example"] = "Choose…"


def _inputs() -> dict:
    s = st.session_state
    return {"gender": s[P + "gender"], "age": float(s[P + "age"]), "hypertension": int(s[P + "hypertension"]),
            "heart_disease": int(s[P + "heart"]), "smoking_history": s[P + "smoking"], "bmi": float(s[P + "bmi"]),
            "HbA1c_level": float(s[P + "hba1c"]), "diabetes": int(s[P + "diabetes"])}


def _estimate() -> None:
    inputs = _inputs()
    try:
        b = data.bundle("regression")
        raw = pd.DataFrame([inputs])[REGRESSION_RAW_COLUMNS]
        pred = float(b["model"].predict(preprocess_regression(raw, b))[0])
        st.session_state[P + "result"] = {"inputs": inputs, "pred": pred, "model": b["model_name"]}
        st.session_state.pop(P + "error", None)
    except Exception as exc:
        st.session_state[P + "error"] = f"{type(exc).__name__}: {exc}"
        st.session_state.pop(P + "result", None)


def render() -> None:
    stats = data.jsonf("regression", "stats.json")
    r = stats["ranges"]
    _init(stats)

    ui.page_header("Glucose Estimate",
                   "A regression model estimates blood glucose from demographic and clinical inputs. "
                   "This page demonstrates the regression track.")

    left, right = st.columns([5, 6], gap="large")
    with left:
        with ui.card("glu-form", "Patient profile", "The regression model uses a different set of inputs from the classifier."):
            ui.section("Demographics")
            c1, c2 = st.columns(2)
            c1.number_input("Age (years)", min_value=float(r["age"]["min"]), max_value=float(r["age"]["max"]), step=1.0,
                            key=P + "age", help=f"Training range {r['age']['min']} to {r['age']['max']:.0f} years.")
            c2.selectbox("Gender", stats["categories"]["gender"], key=P + "gender")
            ui.section("Clinical measurements")
            c3, c4 = st.columns(2)
            c3.number_input("BMI (kg/m²)", min_value=float(r["bmi"]["min"]), max_value=float(r["bmi"]["max"]), step=0.1,
                            key=P + "bmi", help=f"Training range {r['bmi']['min']} to {r['bmi']['max']}.")
            c4.selectbox("Smoking history", [s for s in SMOKING_ORDER if s in stats["categories"]["smoking_history"]],
                         key=P + "smoking", format_func=ui.label)
            st.slider("HbA1c (%)", min_value=float(r["HbA1c_level"]["min"]), max_value=float(r["HbA1c_level"]["max"]),
                      step=0.1, key=P + "hba1c", help="Glycated haemoglobin. Training range 3.5 to 9.0 %.")
            ui.section("Comorbidities and dataset label")
            c5, c6 = st.columns(2)
            c5.toggle("Hypertension", key=P + "hypertension")
            c6.toggle("Heart disease", key=P + "heart")
            st.toggle("Diabetes label (dataset column)", key=P + "diabetes",
                      help="The regression notebook uses the dataset's diabetes label as an input feature. "
                           "It was derived from the same measurements as the target, so it carries much of the model's signal.")
            ui.callout("warn", "About the diabetes label input",
                       f"This column is an input in the notebook's model. Removing it drops linear-model R² from "
                       f"{stats['ablation_r2_linear']['with_diabetes_label']:.2f} to "
                       f"{stats['ablation_r2_linear']['without_diabetes_label']:.2f}.")

            st.button("Estimate Glucose", type="primary", width="stretch", on_click=_estimate, key=P + "go")
            e1, e2 = st.columns(2)
            e1.button("Reset", width="stretch", on_click=_reset, args=(stats,), key=P + "reset")
            e2.selectbox("Load an illustrative example", ["Choose…"] + list(EXAMPLES), key=P + "example",
                         label_visibility="collapsed",
                         on_change=lambda: _load_example(st.session_state[P + "example"], stats)
                         if st.session_state[P + "example"] in EXAMPLES else None)
            ui.caption("Examples are illustrative value sets, not real patients.")

    with right:
        res, err = st.session_state.get(P + "result"), st.session_state.get(P + "error")
        if err:
            with ui.card("glu-error"):
                ui.callout("alert", "The estimate could not be computed", f"{ui.esc(err)}. Check the inputs and try again.")
        elif res is None:
            with ui.card("glu-empty", "Result"):
                ui.empty_state("No estimate yet", "Fill in the profile and press Estimate Glucose, or load an illustrative example.")
                ui.callout("info", "What this model is",
                           f"The best regressor by test R² is <b>{ui.esc(stats['best_model'])}</b>. The target takes only "
                           f"{len(stats['target_distinct_values'])} distinct values, so this is an educational demonstration "
                           f"of regression, not a measurement tool.")
        else:
            _results(res, stats)


def _results(res: dict, stats: dict) -> None:
    inputs, pred = res["inputs"], res["pred"]
    stale = inputs != _inputs()
    bm = stats["best_metrics"]
    sample = data.table("regression", "pred_vs_actual_sample.csv")

    with ui.card("glu-result", "Result"):
        st.markdown(ui.badge(f"Model: {res['model']}", "teal") + " " + ui.badge(f"Test R² {bm['R² Score']:.4f}"),
                    unsafe_allow_html=True)
        if stale:
            ui.callout("warn", "Inputs changed", "The values on the left differ from the ones analysed below. Press Estimate Glucose to update.")
        ui.big_result("Estimated blood glucose (regression output)", f"{pred:.1f}", "mg/dL")
        ui.caption("A model estimate for educational purposes. It is not a measured blood glucose result, "
                   "and no confidence or prediction interval is available for it.")
        ui.section("Model evaluation on the held-out test set")
        ui.tiles([("R²", f"{bm['R² Score']:.3f}", "share of variance explained"),
                  ("RMSE", f"{bm['RMSE']:.1f}", "mg/dL, typical error size"),
                  ("MAE", f"{bm['MAE']:.1f}", "mg/dL, mean absolute error")])
        ui.callout("neutral", "How to read these metrics",
                   f"They describe the model across {stats['test_rows']:,} test patients. They are not a guarantee of the error "
                   f"for this individual estimate. R² of {bm['R² Score']:.2f} means most variation in glucose is not explained by these inputs.")
        st.plotly_chart(charts.pred_vs_actual_fig(sample, marker=pred, model=res["model"]), width="stretch", config=charts.CONFIG)
        ui.caption("The teal line is this estimate. Test-set predictions occupy a narrow band regardless of the true value, "
                   "which is what an R² of 0.18 looks like.")

        with st.expander("Inputs used"):
            ui.kv_table([
                ("Age", f"{inputs['age']:.1f} years"), ("Gender", inputs["gender"]), ("BMI", f"{inputs['bmi']:.1f} kg/m²"),
                ("Smoking history", ui.label(inputs["smoking_history"])), ("HbA1c", f"{inputs['HbA1c_level']:.1f} %"),
                ("Hypertension", "Yes" if inputs["hypertension"] else "No"),
                ("Heart disease", "Yes" if inputs["heart_disease"] else "No"),
                ("Diabetes label (dataset column)", "1" if inputs["diabetes"] else "0"),
                ("Derived: age × BMI", f"{inputs['age'] * inputs['bmi']:.1f}"),
            ])
        out = pd.DataFrame([{**inputs, "predicted_blood_glucose_mg_dl": round(pred, 3), "model": res["model"]}])
        ui.csv_download("Download result (CSV)", out, "glucotrack_glucose_estimate.csv", P + "dl")
