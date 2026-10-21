"""GlucoTrack: Streamlit front end for the 23CSE301 Machine Learning capstone.

Run from the repository root:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="GlucoTrack", page_icon=":material/monitor_heart:", layout="wide",
                   initial_sidebar_state="collapsed")

from glucotrack import data, ui  # noqa: E402
from views import about, comparison, glucose, risk, segment  # noqa: E402

PAGES = [
    st.Page(risk.render, title="Diabetes Risk", url_path="risk", default=True),
    st.Page(glucose.render, title="Glucose Estimate", url_path="glucose"),
    st.Page(segment.render, title="Patient Segment", url_path="segment"),
    st.Page(comparison.render, title="Model Comparison", url_path="compare"),
    st.Page(about.render, title="About", url_path="about"),
]
ui.PAGES = {p.url_path: p for p in PAGES}

navigation = st.navigation(PAGES, position="hidden")
ui.shell(PAGES, active=navigation)

missing = data.missing_artifacts()
if missing:
    ui.callout("alert", "Model artifacts are missing",
               "The app loads files exported by the notebooks and cannot start without them:<br>"
               + "<br>".join(ui.esc(m) for m in missing))
    st.stop()

navigation.run()
