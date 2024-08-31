import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import pandas as pd
from sklearn.exceptions import NotFittedError
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

st.title("Model Testing")
st.write("Last part of this program is to test the 3 models we have on a sample test dataset which contains same variables as the original dataset. We will predict the satisfaction of each passenger in the dataset. ")

col1, col2 = st.columns([1, 3])

with col1:

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        csv_data = pd.read_csv(uploaded_file)
        st.write("CSV data Successfully Read.")

    # Button to trigger display of content in col2
    if st.button("Show Prediction Results"):
        show_results = True
    else:
        show_results = False

with col2:
    # Display prediction results only if the button in col1 is clicked
    if show_results:
        st.subheader("Predictions and Inferencing")

        column_sel = st.selectbox('Select Model', ['Random Forest Classifier','Decision Tree Classifier','Support Vector Machine (SVM)','Logistic Regression'])
        model_sel = column_sel

        if model_sel == 'Random Forest Classifier':

            fl = pd.read_csv("final.csv")
            fl = fl.drop(['id'], axis=1)
            fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
            X = fl.drop('satisfaction', axis=1)
            y = fl['satisfaction'] 

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_train, y_train)
            y_pred = rf_classifier.predict(X_test)

            actual_list = [
                'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'satisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'satisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied'
            ]

            warnings.simplefilter(action='ignore', category=UserWarning)
            new_customer_data = csv_data
            predictions_list = []

            col3, col4 = st.columns(2)

            with col3:
                try:
                    for idx, row in new_customer_data.iterrows():
                        prediction = rf_classifier.predict(row.to_numpy().reshape(1, -1))

                        if prediction[0] == 1:
                            predictions_list.append("satisfied")
                            st.write("Customer is Satisfied")
                        else:
                            predictions_list.append("neutral or dissatisfied")
                            st.write("Customer is Not Satisfied")
                except NotFittedError as e:
                    predictions_list.append("Model has not been fitted yet.")

            with col4:
                c1=0
                c2=0
                for i in range(len(actual_list)):
                    if predictions_list[i] == actual_list[i]:
                        c1+=1
                    else:
                        c2+=1
                
                data = {'Category': ['Correct Predictions', 'Incorrect Predictions'], 'Count': [c1, c2]}
                df = pd.DataFrame(data)
                fig = px.pie(df, values='Count', names='Category', title='Prediction Accuracy')
                st.plotly_chart(fig)


            

        elif model_sel == 'Support Vector Machine (SVM)':

            fl = pd.read_csv("final.csv")
            fl = fl.drop(['id'], axis=1)
            fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
            X= fl.drop('satisfaction', axis=1)
            y= fl['satisfaction']  

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

            svm_model = SVC(kernel='linear', C=1.0, random_state=24)
            svm_model.fit(X_train, y_train)
            kernel_type = svm_model.kernel
            y_pred = svm_model.predict(X_test)

            actual_list = [
                'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'satisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'satisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied'
            ]

            warnings.simplefilter(action='ignore', category=UserWarning)
            new_customer_data = csv_data
            predictions_list = []

            col3, col4 = st.columns(2)

            with col3:
                try:
                    for idx, row in new_customer_data.iterrows():
                        prediction = svm_model.predict(row.to_numpy().reshape(1, -1))

                        if prediction[0] == 1:
                            predictions_list.append("satisfied")
                            st.write("Customer is Satisfied")
                        else:
                            predictions_list.append("neutral or dissatisfied")
                            st.write("Customer is Not Satisfied")
                except NotFittedError as e:
                    predictions_list.append("Model has not been fitted yet.")

            with col4:
                c1=0
                c2=0
                for i in range(len(actual_list)):
                    if predictions_list[i] == actual_list[i]:
                        c1+=1
                    else:
                        c2+=1
                
                data = {'Category': ['Correct Predictions', 'Incorrect Predictions'], 'Count': [c1, c2]}
                df = pd.DataFrame(data)
                fig = px.pie(df, values='Count', names='Category', title='Prediction Accuracy')
                st.plotly_chart(fig)
           
        elif model_sel == 'Decision Tree Classifier':
            
            fl = pd.read_csv("final.csv")
            fl = fl.drop(['id'], axis=1)
            fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
            X = fl.drop('satisfaction', axis=1)
            y = fl['satisfaction'] 

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            param_grid = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            clf = DecisionTreeClassifier()
            grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_

            best_clf = DecisionTreeClassifier(**best_params)

            best_clf.fit(X_train, y_train)

            y_pred = best_clf.predict(X_test)

            actual_list = [
                'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'satisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'satisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied'
            ]

            warnings.simplefilter(action='ignore', category=UserWarning)
            new_customer_data = csv_data
            predictions_list = []

            col3, col4 = st.columns(2)

            with col3:
                try:
                    for idx, row in new_customer_data.iterrows():
                        prediction = best_clf.predict(row.to_numpy().reshape(1, -1))

                        if prediction[0] == 1:
                            predictions_list.append("satisfied")
                            st.write("Customer is Satisfied")
                        else:
                            predictions_list.append("neutral or dissatisfied")
                            st.write("Customer is Not Satisfied")
                except NotFittedError as e:
                    predictions_list.append("Model has not been fitted yet.")

            with col4:
                c1=0
                c2=0
                for i in range(len(actual_list)):
                    if predictions_list[i] == actual_list[i]:
                        c1+=1
                    else:
                        c2+=1
                
                data = {'Category': ['Correct Predictions', 'Incorrect Predictions'], 'Count': [c1, c2]}
                df = pd.DataFrame(data)
                fig = px.pie(df, values='Count', names='Category', title='Prediction Accuracy')
                st.plotly_chart(fig)

        elif model_sel == 'Logistic Regression':
            
            fl = pd.read_csv("final.csv")
            fl = fl.drop(['id'], axis=1)
            fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
            X = fl.drop('satisfaction', axis=1)
            y = fl['satisfaction'] 

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logistic_model = LogisticRegression()
            logistic_model.fit(X_train, y_train)

            y_pred = logistic_model.predict(X_test)

            actual_list = [
                'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'satisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied',
                'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied', 'satisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'neutral or dissatisfied', 'satisfied',
                'satisfied', 'satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied', 'satisfied', 'neutral or dissatisfied'
            ]

            warnings.simplefilter(action='ignore', category=UserWarning)
            new_customer_data = csv_data
            predictions_list = []

            col3, col4 = st.columns(2)

            with col3:
                try:
                    for idx, row in new_customer_data.iterrows():
                        prediction = logistic_model.predict(row.to_numpy().reshape(1, -1))

                        if prediction[0] == 1:
                            predictions_list.append("satisfied")
                            st.write("Customer is Satisfied")
                        else:
                            predictions_list.append("neutral or dissatisfied")
                            st.write("Customer is Not Satisfied")
                except NotFittedError as e:
                    predictions_list.append("Model has not been fitted yet.")

            with col4:
                c1=0
                c2=0
                for i in range(len(actual_list)):
                    if predictions_list[i] == actual_list[i]:
                        c1+=1
                    else:
                        c2+=1
                
                data = {'Category': ['Correct Predictions', 'Incorrect Predictions'], 'Count': [c1, c2]}
                df = pd.DataFrame(data)
                fig = px.pie(df, values='Count', names='Category', title='Prediction Accuracy')
                st.plotly_chart(fig)

         
           