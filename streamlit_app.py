import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Streamlit app title
st.title('Ionising Radiation Experiment Analysis')

# Introduction to the app with code explanation

st.markdown("""
### Introduction
This app guides you through the process of analyzing data from an Ionising Radiation experiment using linear regression. We'll fit a linear model to the data to verify the inverse square law, visualize the fit, and discuss the statistical significance of our results. Below is the code that creates this title and introduction in the app:""")
st.title('Ionising Radiation Experiment Analysis')


# Data upload section with explanation
st.markdown("""
### Data Upload
Upload your CSV data file below. The following code enables file uploading:

uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])""")
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

# Display and process the uploaded file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if not df.empty:
        st.write("First few rows of your dataset:")
        st.dataframe(df.head())
        st.markdown("""
        Here's the code that reads the uploaded CSV and displays the first few rows:
        ```
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        ```
        """)

        # Column selection with explanation
        st.markdown("""
        ### Column Selection
        Select the appropriate columns for distance and counts using dropdown menus. The code for creating these dropdowns is:
        ```
        distance_col = st.selectbox('Select the column for distance (in meters):', df.columns)
        count_col = st.selectbox('Select the column for counts:', df.columns, index=1)
        ```
        """)
        distance_col = st.selectbox('Select the column for distance (in meters):', df.columns)
        count_col = st.selectbox('Select the column for counts:', df.columns, index=1)

        # Data preparation and explanation
        df['Count Rate'] = df[count_col] / 60
        df['Inverse Square Distance'] = 1 / df[distance_col]**2
        st.markdown("""
        ### Data Preparation
        We calculate the count rate and the inverse square of the distance. Here's the code that performs these calculations:
        ```
        df['Count Rate'] = df[count_col] / 60
        df['Inverse Square Distance'] = 1 / df[distance_col]**2
        ```
        """)

        # Linear regression with explanation
        regression_result = linregress(df['Inverse Square Distance'], df['Count Rate'])
        st.markdown("""
        ### Linear Regression
        We apply linear regression to the prepared data. The following code carries out this regression:
        ```
        regression_result = linregress(df['Inverse Square Distance'], df['Count Rate'])
        ```
        """)

        # Visualization and explanation
        fig, ax = plt.subplots()
        # Plot the original data in blue
        ax.plot(df['Inverse Square Distance'], df['Count Rate'], 'o', label='Original data', markersize=10, zorder=2)
        # Plot the regression line
        ax.plot(df['Inverse Square Distance'], regression_result.intercept + regression_result.slope * df['Inverse Square Distance'], 'r', label='Fitted line', zorder=1)
        ax.set_xlabel('Inverse Square of Distance (1/m^2)')
        ax.set_ylabel('Count Rate (counts/s)')
        # Add error bars behind the data points
        error_size = np.sqrt(df[count_col]) / 60  # Increase this if needed for visibility
        ax.errorbar(df['Inverse Square Distance'], df['Count Rate'], yerr=error_size, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=5, zorder=1)
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        ### Visualization
        The plot visualizes the data points and the fitted regression line. Here's how we create this plot:
        ```
        fig, ax = plt.subplots()
        ax.plot(... # plotting commands
        st.pyplot(fig)
        ```
        """)

        # Display and explain regression results
        st.subheader('Regression Analysis')
        # write slope with standard error
        st.write(f"Slope: {regression_result.slope:.4f}", f"± {regression_result.stderr:.4f}")
        st.write(f"Intercept: {regression_result.intercept:.4f}")
        st.write(f"R-squared: {regression_result.rvalue**2:.4f}")
        st.write(f"p-value: {regression_result.pvalue:.4f}")
        st.write(f"Standard error: {regression_result.stderr:.4f}")
        st.markdown("""
        Here's how we display the regression results in the app:
        ```
        st.write(f"Slope: {regression_result.slope:.4f}")
        st.write(f"Intercept: {regression_result.intercept:.4f}")
        st.write(f"R-squared: {regression_result.rvalue**2:.4f}")
        st.write(f"p-value: {regression_result.pvalue:.4f}")
        st.write(f"Standard error: {regression_result.stderr:.4f}")
        ```
        """)

        # Conclusion with code explanation
        if regression_result.rvalue**2 > 0.9:
            st.markdown("### The data strongly supports the inverse square law.")
        else:
            st.markdown("### The data does not strongly support the inverse square law.")
        st.markdown("""
        ### Conclusion
        We provide a conclusion based on the R-squared value. The condition and message are coded as follows:
        ```
        if regression_result.rvalue**2 > 0.9:
            st.markdown("### The data strongly supports the inverse square law.")
        else:
            st.markdown("### The data does not strongly support the inverse square law.")
        ```
        """)
    else:
        st.error("The uploaded file is empty. Please upload a valid CSV file.")
else:
    st.info('Awaiting CSV file to be uploaded for analysis.')
