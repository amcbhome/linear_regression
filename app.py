import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Correlation coefficient
r = np.corrcoef(df['Activity (000s units)'], df['Total Cost (£000)'])[0, 1]

# Page title
st.title("Production Cost Estimator")

# Layout with 3 columns
col1, col2, col3 = st.columns([1, 1, 1])

# Column 1: Regression Plot
with col1:
    st.subheader("Regression Plot")

    # Extended range for line of best fit
    x_range = np.linspace(0, 200, 100).reshape(-1, 1)
    y_range = model.predict(x_range)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['Activity (000s units)'], df['Total Cost (£000)'], color='blue', label='Actual data')
    ax.plot(x_range, y_range, color='red', label='Regression line')

    # If calculated, add forecast point
    if 'activity' in st.session_state and 'predicted' in st.session_state:
        ax.scatter(st.session_state.activity, st.session_state.predicted, color='green', label='Forecast', zorder=5)
        ax.annotate(f"({st.session_state.activity}, £{int(st.session_state.predicted)})",
                    (st.session_state.activity, st.session_state.predicted),
                    textcoords="offset points", xytext=(5,5))

    ax.set_xlim(0, 200)
    ax.set_xlabel('Activity (000s units)')
    ax.set_ylabel('Total Cost (£000)')
    ax.set_title('Linear Regression Forecast')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# Column 2: Data Table
with col2:
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

# Column 3: Forecast calculator
with col3:
    st.subheader("Forecast Calculator")
    activity = st.slider("Select activity level (000s units):", 0, 100, 50)
    if st.button("Calculate"):
        predicted = model.predict(np.array([[activity]]))[0]
        st.session_state.activity = activity
        st.session_state.predicted = predicted
        st.metric(label="Estimated Cost", value=f"£{int(predicted * 1000):,}")
