import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write("This app uses a **Linear Regression** model trained on student data "
             "to predict a **Performance Index** between 10 and 100.")
    st.write("Built with Python, scikit-learn and Streamlit.")
    st.caption("Built by ZAKIR AHMED")

# ---------- Main page ----------
st.title("🎓 Student Performance Predictor")
st.write("Move the sliders to describe a student, then click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    hours = st.slider("📚 Hours studied per day", 1, 9, 5)
    sleep = st.slider("😴 Sleep hours per day", 4, 9, 7)
    papers = st.slider("📝 Sample question papers practiced", 0, 9, 4)

with col2:
    prev = st.slider("🎯 Previous scores", 40, 99, 70)
    extra = st.radio("⚽ Extracurricular activities", ["Yes", "No"], horizontal=True)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    X = pd.DataFrame(
        [[hours, prev, 1 if extra == "Yes" else 0, sleep, papers]],
        columns=["Hours Studied", "Previous Scores", "Extracurricular Activities",
                 "Sleep Hours", "Sample Question Papers Practiced"],
    )
    pred = float(model.predict(X)[0])
    pred = max(0.0, min(100.0, pred))

    st.success("Prediction complete!")
    st.metric("Predicted Performance Index", f"{pred:.1f} / 100")
    st.progress(int(pred))

    if pred >= 75:
        st.balloons()
        st.write("🌟 Excellent expected performance!")
    elif pred >= 50:
        st.write("👍 Good, but there is room to improve.")
    else:
        st.write("⚠️ Needs more study time and practice.")
