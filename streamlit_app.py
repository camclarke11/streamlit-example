import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# App title
st.title('Ionising Radiation Experiment Analysis')

# Markdown text for introduction
st.markdown("""
This app analyses the data from the Ionising Radiation experiment. 
It computes the count rates, estimates errors based on Poisson statistics, and verifies the inverse square law.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # If the dataframe is not empty, display the first few rows and move on to analysis
    if not df.empty:
        st.write("First few rows of your dataset:")
        st.dataframe(df.head())

        # Allow the user to select the columns
        distance_col = st.selectbox('Select the column for distance (in meters):', df.columns)
        count_col = st.selectbox('Select the column for counts:', df.columns)

        # Perform calculations
        df['Count Rate'] = df[count_col] / 60  # Convert to count rate
        df['Error'] = np.sqrt(df[count_col]) / 60  # Estimate error based on Poisson statistics
        df['Inverse Square Distance'] = 1 / df[distance_col]**2
        
        # Perform linear regression on the inverse square of the distance vs count rate
        regression_result = linregress(df['Inverse Square Distance'], df['Count Rate'])

        # Create a plot
        fig, ax = plt.subplots()
        ax.errorbar(df['Inverse Square Distance'], df['Count Rate'], yerr=df['Error'], fmt='o', label='Experimental data')
        ax.plot(df['Inverse Square Distance'], regression_result.intercept + regression_result.slope * df['Inverse Square Distance'], 'r', label='Fit line')
        ax.set_xlabel('Inverse Square of Distance (1/m^2)')
        ax.set_ylabel('Count Rate (counts/s)')
        ax.set_title('Verification of the Inverse Square Law')
        ax.legend()

        # Show regression results and plot
        st.subheader('Regression Analysis')
        st.write(f"Slope (proportional to intensity): {regression_result.slope:.4f}")
        st.write(f"Intercept: {regression_result.intercept:.4f}")
        st.write(f"R-squared value: {regression_result.rvalue**2:.4f}")
        st.write(f"p-value of the regression: {regression_result.pvalue:.4f}")
        st.write(f"Standard error of the estimate: {regression_result.stderr:.4f}")

        st.subheader('Inverse Square Law Fit')
        st.pyplot(fig)

        # Conclusion based on R-squared value
        if regression_result.rvalue**2 > 0.9:
            st.markdown("### The data strongly supports the inverse square law.")
        else:
            st.markdown("### The data does not strongly support the inverse square law.")

    else:
        st.error("The uploaded file is empty. Please upload a valid CSV file.")
else:
    st.info('Awaiting CSV file to be uploaded for analysis.')
