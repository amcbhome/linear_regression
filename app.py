import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
data = {
    'Activity (000s units)': [15, 45, 25, 55, 30, 20, 35, 60],
    'Total Cost ($000)':     [300, 615, 470, 680, 520, 350, 590, 740]
}
df = pd.DataFrame(data)

# Fit model
X = df[['Activity (000s units)']]
y = df['Total Cost ($000)']
model = LinearRegression()
model.fit(X, y)

# Correlation and R-squared
r = np.corrcoef(df['Activity (000s units)'], df['Total Cost ($000)'])[0, 1]
r_squared = model.score(X, y)

# App
st.title("Production Cost Estimator")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Based on historical Data (r = {r:.2f})")
    st.table(df)

with col2:
    st.subheader("Forecast Calculator")
    activity = st.slider("Select activity level (000s units):", 0, 100, 50)
    if st.button("Calculate"):
        predicted = model.predict(np.array([[activity]]))[0]
        st.metric(label="Estimated Cost", value=f"£{int(predicted * 1000):,}")
