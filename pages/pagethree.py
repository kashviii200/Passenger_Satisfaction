import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

st.title("Model Selection")
st.write("Now, we split the prepared data into training and testing batches and fit the data in the 3 models mentioned in the start. A drop down menu is also provided for selecting which model to run. On selecting a particular model, the program will display various metrics and charts showing how well the data is being fit into the model. ")

col1, col2 = st.columns(2)

with col1:

    fl = pd.read_csv("final.csv")

    X = fl.drop('satisfaction', axis=1)
    y = fl['satisfaction']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    column_sel = st.selectbox('Select Model', ['Random Forest Classifier','Decision Tree Classifier','Support Vector Machine (SVM)', 'K-Nearest Neighbor', 'Logistic Regression'])
    model_sel = column_sel

    if model_sel == 'Random Forest Classifier':
        st.subheader("About")
        st.write("The main idea behind Random Forest is to build a large number of decision trees during training and then combine their predictions to make the final prediction.")
        st.write(" - Random Sampling: Each tree in the Random Forest is trained on a random subset of the training data (bootstrapping) and a random subset of features, adding diversity to the ensemble.")
        st.write(' - Feature Importance: Random Forest provides a measure of feature importance, allowing you to understand which features are most relevant for making predictions.')

    elif model_sel == 'Support Vector Machine (SVM)' : 
        st.subheader("About")
        st.write("SVM aims to find the hyperplane that maximizes the margin between different classes in the feature space. The margin is the distance between the hyperplane and the nearest data point from each class, known as support vectors.")
        st.write(" - Kernel Trick: SVM uses the kernel trick to implicitly map the input features into higher-dimensional spaces. This allows SVM to handle non-linear decision boundaries and separate non-linearly separable classes.")
    
    elif model_sel == 'Decision Tree Classifier':
        st.subheader("About")
        st.write("It creates a tree-like structure where each internal node represents a feature or attribute, each branch represents a decision rule based on that feature, and each leaf node represents the outcome or class label.")
        st.write(" - Non-Parametric: Decision trees are non-parametric models, meaning they do not make any assumptions about the underlying data distribution. They can handle both numerical and categorical data without the need for feature scaling.")
        st.write(" - Handles Nonlinear Relationships: Decision trees can capture complex nonlinear relationships between features and target variables. They recursively split the data based on thresholds to create regions or segments with different class labels.")

with col2:
    
    if model_sel == 'Random Forest Classifier':

        fl = pd.read_csv("final.csv")

        fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
        X = fl.drop('satisfaction', axis=1)
        y = fl['satisfaction']            

        st.write(" ")
        st.write(" ")
        st.write("Train % : 80% & Test : 20%")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy of model")
        st.subheader(accuracy)
        st.write("Confusion Matrix")
        fig = px.imshow(confusion_matrix(y_test, y_pred),
                labels=dict(x="Predicted Labels", y="True Labels"),
                x=['Not Satisfied', 'Satisfied'], y=['Not Satisfied', 'Satisfied'],
                color_continuous_scale='Blues')
        fig.update_layout(width=500, height=500)
        st.plotly_chart(fig)
        

        st.write("Classification Report")
        st.code(classification_report(y_test, y_pred), language='text')
        st.write("Class 0: It has a precision, recall, and F1-score of 0.95, 0.97, and 0.96 respectively. This indicates the model performs very well in identifying and classifying class 0 instances. It rarely makes mistakes (high precision) and catches most of the actual class 0 examples (high recall). There are 2945 instances of class 0 in the data.")
        st.write("Class 1: Similar to class 0, the model performs well with a precision, recall, and F1-score of 0.95, 0.94, and 0.95 respectively. There are 2251 instances of class 1 in the data.")


        fpr, tpr, _ = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:,1])
        roc_curve_plot = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve')
        random_guess = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess')
        fig2 = go.Figure(data=[roc_curve_plot, random_guess])
        fig2.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig2)
        st.write("The ROC curve in the image appears to be above the random guess line, which suggests that the model is performing better than random guessing.")


        precision, recall, _ = precision_recall_curve(y_test, rf_classifier.predict_proba(X_test)[:,1])
        pr_curve_plot = go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve')
        fig3 = go.Figure(data=pr_curve_plot)
        fig3.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(fig3)
        st.write("This suggests the model prioritizes precision at the beginning, correctly classifying most positive predictions but potentially missing some positive cases. As the threshold loosens to capture more positive cases, the precision slightly falters.")

        selected_tree = DecisionTreeClassifier(max_depth=3)
        selected_tree.fit(X_train, y_train)


        feature_importances = rf_classifier.feature_importances_
        feature_names = X_train.columns.tolist()

        # Sort the feature importances and feature names by importance values
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_feature_importances = feature_importances[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # Create a bar chart for feature importances using Plotly
        fig5 = go.Figure(go.Bar(
            x=sorted_feature_importances,
            y=sorted_feature_names,
            orientation='h',
            marker_color='royalblue',  # Optional: Set color of bars
        ))
        fig5.update_layout(title='Feature Importances',
                        xaxis_title='Importance',
                        yaxis_title='Features')
        st.plotly_chart(fig5)  
        st.write("Online Boarding, Business Class and Buisness travel type are the most important features in Random forest Classifier.")


    elif model_sel == 'Support Vector Machine (SVM)':

        fl = pd.read_csv("final.csv")

        fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
        X = fl.drop('satisfaction', axis=1)
        y = fl['satisfaction']   

        st.write(" ")
        st.write(" ")
        st.write("Train % : 80% & Test : 20%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

        svm_model = SVC(kernel='linear', C=1.0, random_state=24)
        
        svm_model.fit(X_train, y_train)
        kernel_type = svm_model.kernel

        y_pred = svm_model.predict(X_test)
        st.write("Accuracy of model")
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader(accuracy)

        st.write("Classification Report")
        st.code(classification_report(y_test, y_pred), language='text')

        st.write("Class 0: It has a precision of 0.87, recall of 0.92, and F1-score of 0.90. This indicates the model performs well in identifying and classifying class 0 instances. It rarely makes mistakes (high precision) and catches most of the actual class 0 examples (high recall). There are 2872 instances of class 0 in the data.")
        st.write("Class 1: It has a precision of 0.89, recall of 0.84, and F1-score of 0.86. The model performs well with a similar pattern to class 0, but with slightly lower recall. There are 2324 instances of class 1 in the data.")

        classes = ['Neutral or Dissatisfied', 'Satisfied']
        precision = [precision_score(y_test, y_pred, average=None)[0], precision_score(y_test, y_pred, average=None)[1]]
        recall = [recall_score(y_test, y_pred, average=None)[0], recall_score(y_test, y_pred, average=None)[1]]
        f1_score = [f1_score(y_test, y_pred, average=None)[0], f1_score(y_test, y_pred, average=None)[1]]

        # Create a bar chart for precision, recall, and F1-score
        fig = go.Figure(data=[
            go.Bar(name='Precision', x=classes, y=precision),
            go.Bar(name='Recall', x=classes, y=recall),
            go.Bar(name='F1-score', x=classes, y=f1_score)
        ])

        # Update layout
        fig.update_layout(title='Classification Report',
                        xaxis_title='Class',
                        yaxis_title='Score',
                        barmode='group')

        # Show plot using Streamlit's plotly_chart function
        st.plotly_chart(fig)

        st.write("The model performs slightly better in recalling actual class 0 instances (0.92 recall) compared to class 1 instances (0.86 recall). This suggests the model might miss a few relevant class 1 cases.")

        st.write("Confusion Matrix")
        fig = px.imshow(confusion_matrix(y_test, y_pred),
                labels=dict(x="Predicted Labels", y="True Labels"),
                x=['Not Satisfied', 'Satisfied'], y=['Not Satisfied', 'Satisfied'],
                color_continuous_scale='Blues')
        fig.update_layout(width=500, height=500)
        st.plotly_chart(fig)
        

        if kernel_type == 'linear':
            coefficients = svm_model.coef_[0]  # Extract coefficients for the first class (assuming binary classification)
            feature_names = X.columns  # Assuming X is your feature matrix

            # Sort feature names and coefficients by absolute coefficient values
            sorted_indices = np.argsort(np.abs(coefficients))[::-1]
            sorted_coefficients = coefficients[sorted_indices]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]

            # Create horizontal bar plot for feature importance using Plotly
            fig = go.Figure(go.Bar(
                x=sorted_coefficients,
                y=sorted_feature_names,
                orientation='h',
            ))
            fig.update_layout(title='Feature Importance for Linear SVM',
                            xaxis_title='Coefficient Magnitude',
                            yaxis_title='Feature')
            
            # Show plot using Streamlit's plotly_chart function
            st.plotly_chart(fig)
        else:
            st.write("Model is not using a linear kernel.")

        st.write("Shows both negative and postive importances of features. Business Travel being the feature with most positive importance, and Personal Travel being the feautre with most negative importance")

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        fig = px.line(x=fpr, y=tpr, title='ROC Curve',
              labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='darkorange', dash='dash'), showlegend=False))
        fig.update_layout(xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))

        # Display the plot using Streamlit's plotly_chart function
        st.plotly_chart(fig)

        st.write("The curve is not close to the top left corner, showing its comparatively low accuracy.")

        precision, recall, _ = precision_recall_curve(y_test, y_pred)

        fig = px.line(x=recall, y=precision, title='Precision-Recall Curve',
                    labels={'x': 'Recall', 'y': 'Precision'})
        fig.update_layout(xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))

        # Display the plot using Streamlit's plotly_chart function
        st.plotly_chart(fig)

        st.write("High precision value and low recall at the start is a good performing model.")
        feature1 = 'Seat comfort'
        feature2 = 'Inflight entertainment'

        # Extract selected features and target variable
        X_selected = fl[[feature1, feature2]]
        y_selected = fl['satisfaction']

        # Train SVM model
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_selected, y_selected)

        # Create meshgrid for decision boundary visualization
        x_min, x_max = X_selected[feature1].min() - 1, X_selected[feature1].max() + 1
        y_min, y_max = X_selected[feature2].min() - 1, X_selected[feature2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and data points
        fig = px.scatter(x=X_selected[feature1], y=X_selected[feature2], color=y_selected)
        fig.add_contour(x=np.arange(x_min, x_max, 0.1),
                        y=np.arange(y_min, y_max, 0.1),
                        z=Z,
                        colorscale='Viridis',
                        opacity=0.5,
                        showscale=False)
        fig.update_layout(title='SVM Decision Boundary Visualization',
                        xaxis_title=feature1,
                        yaxis_title=feature2)

        # Display the plot using Streamlit's plotly_chart function
        st.plotly_chart(fig)    
        st.write("The calibration curve seems to be close to a diagonal line, which suggests the model is well-calibrated. This means the model's predicted probabilities are reliable indicators of the true probability of an event.")

    elif model_sel == 'Decision Tree Classifier':

        fl = pd.read_csv("final.csv")

        fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
        X = fl.drop('satisfaction', axis=1)
        y = fl['satisfaction']   

        st.write(" ")
        st.write(" ")
        st.write("Train % : 80% & Test : 20%")

        param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }

        clf = DecisionTreeClassifier()

        # Initializing GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

        grid_search.fit(X_train, y_train)

        # Getting the best parameters
        best_params = grid_search.best_params_

        best_clf = DecisionTreeClassifier(**best_params)

        best_clf.fit(X_train, y_train)

        y_pred = best_clf.predict(X_test)

        st.write("Accuracy of model")
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader(accuracy)

        clf = DecisionTreeClassifier()
        clf.fit(X, y)


        feature_importance = clf.feature_importances_
        feature_names = X.columns
        feature_df = pd.DataFrame({'Feature Importance': feature_importance, 'Features': feature_names})
        fig = px.bar(feature_df, x='Feature Importance', y='Features', orientation='h',
                    labels={'x': 'Feature Importance', 'y': 'Features'}, title='Feature Importance Plot')
        st.plotly_chart(fig)
        st.write("Online boarding, Inflight wifi service and Business type of travel are the most important features in Decision Tree Classifier.")

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['True Negative', 'True Positive'], columns=['Predicted Negative', 'Predicted Positive'])

        # Plot the confusion matrix using Plotly Express
        fig = px.imshow(cm_df, 
                        labels=dict(x="Predicted", y="True", color="Counts"), 
                        x=cm_df.columns, 
                        y=cm_df.index)
        fig.update_layout(title='Confusion Matrix')

        # Display the plot using Streamlit
        st.plotly_chart(fig)


        y_test_binary = y_test.replace({'neutral or dissatisfied': 0, 'satisfied': 1})

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test_binary, best_clf.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Create the ROC curve plot using Plotly
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC Curve (AUC = {roc_auc:.2f})',
                                line=dict(color='blue', width=2)))
        fig2.add_shape(type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=0, y1=1)

        # Update layout and axis labels
        fig2.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate')

        # Show the plot using Streamlit
        st.plotly_chart(fig2)
        st.write("ROC curve suggests that the model is performing well at distinguishing between positive and negative cases.")

        y_test_binary = y_test.replace({'neutral or dissatisfied': 0, 'satisfied': 1})

        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test_binary, best_clf.predict_proba(X_test)[:, 1])

        # Create the Precision-Recall curve trace
        pr_curve = go.Scatter(x=recall, y=precision, mode='lines', line=dict(color='blue', width=2), name='Precision-Recall curve')

        # Create layout
        layout = go.Layout(
            title='Precision-Recall Curve',
            xaxis=dict(title='Recall'),
            yaxis=dict(title='Precision'),
            legend=dict(x=0, y=1, traceorder="normal")
        )

        # Create figure
        fig3 = go.Figure(data=[pr_curve], layout=layout)

        # Show the plot using Streamlit
        st.plotly_chart(fig3)
        st.write("The precision-recall curve appears to be a good balance between precision and recall, starting at a high precision value and gradually decreasing as recall increases. This suggests the model prioritizes precision at first, correctly classifying most positive predictions, and then focuses on capturing more positive cases as the threshold loosens, although with a slight decrease in precision.")


        def partial_dependence(clf, X_train, feature_index, grid_resolution=100):
            feature_values = np.linspace(np.min(X_train[:, feature_index]), np.max(X_train[:, feature_index]), grid_resolution)
            averaged_predictions = []
            for value in feature_values:
                X_temp = X_train.copy()
                X_temp[:, feature_index] = value
                predictions = clf.predict_proba(X_temp)[:, 1]
                averaged_predictions.append(np.mean(predictions))
            return feature_values, np.array(averaged_predictions)

        # Assuming X_train, best_clf, and X.columns are defined

        # Define PDP features
        pdp_features = [2]  # Choose feature indices for PDP

        # Create a Plotly figure
        fig4 = go.Figure()

        # Iterate over PDP features and add traces to the figure
        for feature in pdp_features:
            p, pdp = partial_dependence(best_clf, X_train.values, feature)
            fig4.add_trace(go.Scatter(x=p, y=pdp, mode='lines', name=X.columns[feature]))

        # Update layout and show the plot using Streamlit
        fig4.update_layout(title='Partial Dependence Plots', xaxis_title='Feature Values', yaxis_title='Partial Dependence')
        st.plotly_chart(fig4)
        st.write("This plot shows the dependance of values of the 2 features ")

        st.write("We are choosing the feature Inflight Wifi Service")
        st.write("Each PDP shows how the predicted outcome changes as the value of a specific feature varies, while keeping other features constant or at their average values.")
        st.write("If the PDP line slopes upward, it suggests a positive relationship between the feature and the predicted outcome. As the feature value increases, the predicted outcome also tends to increase.")

    elif model_sel == 'K-Nearest Neighbor':

        fl = pd.read_csv("final.csv")

        fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
        X = fl.drop('satisfaction', axis=1)
        y = fl['satisfaction']   

        st.write(" ")
        st.write(" ")
        st.write("Train % : 80% & Test : 20%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    elif model_sel == 'Logistic Regression':
        
        fl = pd.read_csv("final.csv")

        fl['satisfaction'] = fl['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1}) 
        X = fl.drop('satisfaction', axis=1)
        y = fl['satisfaction']   

        st.write(" ")
        st.write(" ")
        st.write("Train % : 80% & Test : 20%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)

        y_pred = logistic_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.subheader(accuracy)      

        st.code(classification_report(y_test, y_pred), language='text')

        classes = ['Neutral or Dissatisfied', 'Satisfied']
        precision = [precision_score(y_test, y_pred, average=None)[0], precision_score(y_test, y_pred, average=None)[1]]
        recall = [recall_score(y_test, y_pred, average=None)[0], recall_score(y_test, y_pred, average=None)[1]]
        f1_score_vals = [f1_score(y_test, y_pred, average=None)[0], f1_score(y_test, y_pred, average=None)[1]]

        fig = go.Figure(data=[
            go.Bar(name='Precision', x=classes, y=precision),
            go.Bar(name='Recall', x=classes, y=recall),
            go.Bar(name='F1-score', x=classes, y=f1_score_vals)
        ])

        fig.update_layout(title='Classification Report',
                        xaxis_title='Class',
                        yaxis_title='Score',
                        barmode='group')

        st.plotly_chart(fig)

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Neutral or Dissatisfied', 'Satisfied'], columns=['Neutral or Dissatisfied', 'Satisfied'])


        fig = px.imshow(cm_df, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Neutral or Dissatisfied', 'Satisfied'],
                        y=['Neutral or Dissatisfied', 'Satisfied'],
                        title="Confusion Matrix",
                        color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_xaxes(side="top")

        st.plotly_chart(fig)

        y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line_dash='dash', name='Random Guess'))
        fig.update_layout(title='ROC Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate')

        st.plotly_chart(fig)


        coefficients = logistic_model.coef_[0]
        feature_names = X.columns
        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        sorted_coefficients = coefficients[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]
        

        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]
        sorted_coefficients = [coefficients[i] for i in sorted_indices]

        fig = go.Figure(go.Bar(
            x=sorted_coefficients,
            y=sorted_feature_names,
            orientation='h',
            marker=dict(color=np.sign(sorted_coefficients), colorscale='RdBu', line=dict(color='black', width=1)),
        ))
        fig.update_layout(title='Feature Importance for Logistic Regression',
                        xaxis_title='Coefficient Magnitude',
                        yaxis_title='Feature')

        st.plotly_chart(fig)


if st.button("Model Inferencing and Results"):
    switch_page("pagefour")

if st.button("Go to Previous Page"):
    switch_page("pagetwo")