import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
data = {
    'Activity (000s units)': [15, 45, 25, 55, 30, 20, 35, 60],
    'Total Cost (£000)':     [300, 615, 470, 680, 520, 350, 590, 740]
}
df = pd.DataFrame(data)

# Fit model
X = df[['Activity (000s units)']]
y = df['Total Cost (£000)']
model = LinearRegression()
model.fit(X, y)

# Correlation and R-squared
r = np.corrcoef(df['Activity (000s units)'], df['Total Cost (£000)'])[0, 1]
r_squared = model.score(X, y)

# App layout
st.title("Production Cost Estimator")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Based on historical Data (r = {r:.2f})")

    html_table = """
    <style>
    table {
        margin-left: auto;
        margin-right: auto;
        text-align: center;
        border-collapse: collapse;
    }
    th, td {
        padding: 8px 16px;
        text-align: center;
        border: 1px solid #ccc;
    }
    </style>
    <table>
        <thead>
            <tr>
                <th>Index</th>
                <th>Activity (000s units)</th>
                <th>Total Cost (£000)</th>
            </tr>
        </thead>
        <tbody>
    """

    for idx, row in df.iterrows():
        html_table += f"<tr><td>{idx}</td><td>{row['Activity (000s units)']}</td><td>£{row['Total Cost (£000)']}</td></tr>"

    html_table += """
        </tbody>
    </table>
    <br>
    <div style="text-align: center; font-size: 14px;">
        <a href="https://www.accaglobal.com/uk/en/student/exam-support-resources/fundamentals-exams-study-resources/f5/technical-articles/regression.html" target="_blank">
            Source: ACCA Technical Article on Regression
        </a>
    </div>
    """

    st.markdown(html_table, unsafe_allow_html=True)

with col2:
    st.subheader("Forecast Calculator")
    activity = st.slider("Select activity level (000s units):", 0, 100, 50)
    if st.button("Calculate"):
        predicted = model.predict(np.array([[activity]]))[0]
        st.metric(label="Estimated Cost", value=f"£{int(predicted * 1000):,}")

