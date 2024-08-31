import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


st.title("Cleaning and Visualizing")

st.write("Here, the data is cleaned and visualized. Steps such as Removing Missing values, Outliers, One Hot Encoding are performed to prepare the data for modelling.")

col1, col2 = st.columns(2)

with col1:

    fl = pd.read_csv("test.csv")
    st.write("First, to clean the data, we will split the entire dataset into Numeric and Categorical Columns. Along with this we will also drop the column 'Unnamed'.")
                 
    X = fl.drop('satisfaction', axis=1)
    y = fl['satisfaction']

    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    columns1_df = pd.DataFrame({'Numerical Columns': numerical_cols})
    columns2_df = pd.DataFrame({'Categorical Columns': categorical_cols})

    col3, col4 = st.columns(2)

    with col3:
        st.write(columns1_df)
    with col4:
        st.write(columns2_df)


with col2: 

    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

    X_num_transformed = numerical_transformer.fit_transform(X[numerical_cols])
    st.write("Standardizing the numeric data using StandardScaler and handling missing values with Imputer")
    X_num_df = pd.DataFrame(X_num_transformed, columns=numerical_cols)
    st.write(X_num_df)

    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])      

    X_cat_transformed = categorical_transformer.fit_transform(X[categorical_cols])
    st.write("Performing One Hot Encoding")
    X_cat_df = pd.DataFrame(X_cat_transformed.toarray(), columns=categorical_transformer['encoder'].get_feature_names(categorical_cols))
    st.write(X_cat_df)

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

st.write("Transformed Dataset")
processed_data = pd.concat([X_num_df, X_cat_df], axis=1)
st.write(processed_data)

col5, col6 = st.columns(2)

with col5:

    processed_data['Z-Score'] = (processed_data['Arrival Delay in Minutes'] - processed_data['Arrival Delay in Minutes'].mean()) / processed_data['Arrival Delay in Minutes'].std()

    # Define a threshold for z-score (e.g., z-score greater than 3 or less than -3)
    z_score_threshold = 0.3

    # Filter rows based on z-score to remove outliers
    processed_data_without_outliers = processed_data[(processed_data['Z-Score'] <= z_score_threshold) & (processed_data['Z-Score'] >= -z_score_threshold)]

    # Create a box plot using Plotly Express
    fig = px.box(processed_data_without_outliers, y='Arrival Delay in Minutes', title='Box Plot of Arrival Delay (Processed Data without Outliers)')
    fig.update_xaxes(title_text='Arrival Delay in Minutes')
    st.plotly_chart(fig)


    missing_data_heatmap = px.imshow(processed_data.isnull(), title='Missing Values Heatmap')
    st.plotly_chart(missing_data_heatmap)


with col6:

    processed_data['Z-Score'] = (processed_data['Departure Delay in Minutes'] - processed_data['Departure Delay in Minutes'].mean()) / processed_data['Departure Delay in Minutes'].std()

    # Define a threshold for z-score (e.g., z-score greater than 3 or less than -3)
    z_score_threshold = 0.3

    # Filter rows based on z-score to remove outliers
    processed_data_without_outliers = processed_data[(processed_data['Z-Score'] <= z_score_threshold) & (processed_data['Z-Score'] >= -z_score_threshold)]

    # Create a box plot using Plotly Express
    fig = px.box(processed_data_without_outliers, y='Departure Delay in Minutes', title='Box Plot of Departure Delay (Processed Data without Outliers)')
    fig.update_xaxes(title_text='Departure Delay in Minutes')
    st.plotly_chart(fig)

    st.write("Scatter Plot of Delay")
    x_column = processed_data['Departure Delay in Minutes']  # Replace with the name of your x-axis column
    y_column = processed_data['Arrival Delay in Minutes']
    fig3 = px.scatter(processed_data, x=x_column, y=y_column)
    st.plotly_chart(fig3)


if st.button("Data Modelling and Model Selection"):
    switch_page("pagethree")

if st.button("Go to Previous Page"):
    switch_page("pageone")


    