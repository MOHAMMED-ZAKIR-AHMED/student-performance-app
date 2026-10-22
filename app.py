import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

FEATURES = ["Hours Studied", "Previous Scores", "Extracurricular Activities",
            "Sleep Hours", "Sample Question Papers Practiced"]

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write("This app uses a **Linear Regression** model trained on student data "
             "to predict a **Performance Index** between 10 and 100.")
    st.write("Built with Python, scikit-learn and Streamlit.")
    st.caption("Built by ZAKIR AHMED")

# ---------- Title and inputs ----------
st.title("🎓 Student Performance Predictor")
st.write("Describe a student, then explore the prediction and how the model works.")

col1, col2 = st.columns(2)
with col1:
    hours = st.slider("📚 Hours studied per day", 1, 9, 5)
    sleep = st.slider("😴 Sleep hours per day", 4, 9, 7)
    papers = st.slider("📝 Sample question papers practiced", 0, 9, 4)
with col2:
    prev = st.slider("🎯 Previous scores", 40, 99, 70)
    extra = st.radio("⚽ Extracurricular activities", ["Yes", "No"], horizontal=True)

X = pd.DataFrame(
    [[hours, prev, 1 if extra == "Yes" else 0, sleep, papers]],
    columns=FEATURES,
)

st.divider()

tab1, tab2 = st.tabs(["🔮 Prediction", "🧠 How it works"])

# ---------- Tab 1: Prediction ----------
with tab1:
    if st.button("Predict", type="primary", use_container_width=True):
        pred = float(np.ravel(model.predict(X))[0])
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
    else:
        st.info("Click **Predict** to see the result. Then open the **How it works** tab.")

# ---------- Tab 2: How it works ----------
with tab2:
    st.subheader("The big picture")
    st.markdown(
        """
        1. **Input:** you enter 5 details about the student.
        2. **Weights:** during training, the model learned one *weight* for each input.
           A weight says how many points the Performance Index changes when that input goes up by 1.
        3. **Add up:** the model multiplies each input by its weight, adds them together,
           and adds a starting value (the *intercept*).
        4. **Result:** the total is the predicted Performance Index.
        """
    )

    st.latex(r"\text{Index} = b + w_1 \cdot hours + w_2 \cdot previous + w_3 \cdot extra + w_4 \cdot sleep + w_5 \cdot papers")

    coefs = np.ravel(model.coef_)
    intercept = float(np.ravel(model.intercept_)[0])

    st.subheader("1. What the model learned")
    st.write("These weights were learned in Colab from real student data. "
             "A **positive** weight raises the score, a **negative** weight lowers it.")

    weights_df = pd.DataFrame({
        "Feature": FEATURES,
        "Weight": np.round(coefs, 4),
        "Meaning": [f"+1 adds {c:.2f} points" if c > 0 else f"+1 changes score by {c:.2f} points"
                    for c in coefs],
    })
    st.dataframe(weights_df, hide_index=True, use_container_width=True)
    st.caption(f"Starting value (intercept) b = {intercept:.4f}")

    st.subheader("2. Calculation for the student you selected")
    values = X.iloc[0].values
    contrib = coefs * values

    calc_df = pd.DataFrame({
        "Feature": FEATURES,
        "Your input": values,
        "Weight": np.round(coefs, 4),
        "Contribution (input × weight)": np.round(contrib, 3),
    })
    st.dataframe(calc_df, hide_index=True, use_container_width=True)

    raw = intercept + contrib.sum()
    final = max(0.0, min(100.0, raw))

    c1, c2 = st.columns(2)
    c1.metric("Sum of all parts", f"{raw:.2f}")
    c2.metric("Final Performance Index", f"{final:.1f}")
    st.caption("The final value is kept between 0 and 100. It matches the Prediction tab.")

    st.subheader("3. What pushed the score up or down")
    chart_df = pd.DataFrame(
        {"Contribution": np.append(contrib, intercept)},
        index=FEATURES + ["Starting value (b)"],
    )
    st.bar_chart(chart_df)
    st.caption("Bigger bars mean that input has a bigger effect on the predicted score.")

    st.subheader("About the model")
    st.markdown(
        """
        - **Algorithm:** Linear Regression (scikit-learn)
        - **Goal:** predict the Performance Index (a number from 10 to 100)
        - **Inputs:** hours studied, previous scores, extracurricular activities,
          sleep hours, sample papers practiced
        - **Limits:** linear regression assumes each input has a straight-line effect,
          so it is a simple model for learning purposes, not a perfect predictor.
        """
    )

# ---------- Footer ----------
st.divider()
st.caption("Built by ZAKIR AHMED | [GitHub](https://github.com/MOHAMMED-ZAKIR-AHMED) | "
           "[LinkedIn](https://www.linkedin.com/in/zakir00) | Python, scikit-learn and Streamlit")
