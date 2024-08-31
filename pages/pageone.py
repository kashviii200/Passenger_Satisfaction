import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import pandas as pd
import numpy as np
import plotly.express as px


def page_one():

    col1, col2 = st.columns(2)

    with col1:
        st.title("Understanding the Raw Data")
        st.write("Understand the data which is going to be used to classification. You can see the unclean data columns and their description. Further, there are visualizations as well which help in understand the raw data.")

        fl = pd.read_csv("test.csv")

        st. write("  ")      
        st. write("Number of records before cleaning")
        st.subheader(fl.shape[0])

        st. write("  ")
        st.write("Description of Dataset")
        st.write(fl.describe())

        st.write("Number of Null Values")
        st.write(fl.isna().sum())

        st.title(" ")
        st.title(" ")
        st.title(" ")
        st.title(" ")
        
        dep_delay = fl['Departure Delay in Minutes']
        arr_delay = fl['Arrival Delay in Minutes']

        # Create DataFrames for each box plot
        df_dep = pd.DataFrame({'Delay in Minutes': dep_delay, 'Type': 'Departure Delay'})
        df_arr = pd.DataFrame({'Delay in Minutes': arr_delay, 'Type': 'Arrival Delay'})

        # Concatenate the DataFrames to combine the data for both box plots
        combined_df = pd.concat([df_dep, df_arr])

        # Create the box plot using Plotly Express
        fig = px.box(combined_df, x='Type', y='Delay in Minutes', title='Box Plots of Departure and Arrival Delays',
                    height=900, width=800)

        # Display the plot on Streamlit
        st.plotly_chart(fig)

    with col2:

        st.title(" ")
        st.title(" ")
        st.title(" ")
        st.title(" ")
        st.title(" ")
        st.write("Visualizing Categorical Data")

        column_sel = st.selectbox('Select Categorical Column', ['Gender','Customer Type','Type of Travel','Class'])
        categorical_column = column_sel

        plot_sel = st.selectbox('Select Plot', ['Bar Chart','Pie Chart'])
        if plot_sel == 'Bar Chart':
            selected_chart = px.bar(fl[categorical_column].value_counts().reset_index(), x=fl.index, y=categorical_column, title='Value Counts for Categorical Column')
            st.plotly_chart(selected_chart)    
        elif plot_sel == 'Pie Chart':
            selected_chart = px.pie(fl[categorical_column].value_counts().reset_index(), values=categorical_column, names='index', title='Value Counts for Categorical Column')
            st.plotly_chart(selected_chart)


        missing_data_heatmap = px.imshow(fl.isnull(), title='Missing Values Heatmap')
        st.plotly_chart(missing_data_heatmap)


    if st.button("Data Cleaning and Visualization"):
        switch_page("pagetwo")

    if st.button("Go to Previous Page"):
        switch_page("homepage")

if __name__ == '__main__':
    page_one()