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

# Correlation
r = np.corrcoef(df['Activity (000s units)'], df['Total Cost (£000)'])[0, 1]

# App layout
st.title("Production Cost Estimator")

col1, col2, col3 = st.columns([1, 1, 1])  # Three equal columns

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

with col3:
    st.subheader("Forecast Calculator")
    activity = st.slider("Select activity level (000s units):", 0, 100, 50)
    show_plot = st.button("Calculate")

    if show_plot:
        predicted = model.predict(np.array([[activity]]))[0]
        st.metric(label="Estimated Cost", value=f"£{int(predicted * 1000):,}")

with col1:
    if show_plot:
        st.subheader("Regression Plot")

        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(df['Activity (000s units)'], df['Total Cost (£000)'], color='blue', label='Actual data')
        ax.plot(df['Activity (000s units)'], model.predict(X), color='red', label='Regression line')

        # Predicted point
        ax.scatter(activity, predicted, color='green', label='Forecast', zorder=5)
        ax.annotate(f"({activity}, £{int(predicted)})", (activity, predicted), textcoords="offset points", xytext=(5,5))

        ax.set_xlabel('Activity (000s units)')
        ax.set_ylabel('Total Cost (£000)')
        ax.set_title('Linear Regression Forecast')
        ax.legend()
        st.pyplot(fig)
