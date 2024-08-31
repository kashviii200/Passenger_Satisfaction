import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.title("Passenger Satisfaction Predictor")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Project Details")
    # Text input for project details
    st.write('''The following program 
                is built for predicting whether passegers flying on 
                airlines are Satisfied or Dissatisfied with their experience flying on that particular airline. 
                The entire problem statement demands a Classification approach, which is why we chose 3 models:''')
    st.write("- Random Forest")
    st.write("- SVM")
    st.write("- Decision Tree")
    st.write("to test our data and classify passengers.")

with col2:

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        csv_data = pd.read_csv(uploaded_file)
        st.write("CSV data Successfully Read.")

if st.button("Raw Data Visualization"):
    switch_page("pageone")