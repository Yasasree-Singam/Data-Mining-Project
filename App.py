import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from data_cleaning import clean_data
from classification import train_random_forest, train_svm, train_knn, plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocess_data
import joblib
from sklearn.preprocessing import LabelEncoder

# Function to preprocess user input data
def preprocess_user_input(user_input_df, transformer):
    # Apply the same preprocessing steps as done for training data using the provided transformer
    user_input_processed = transformer.transform(user_input_df)

    return user_input_processed


# Load and clean data
data_balance,crime = clean_data()

def page1():
    st.title("Los Angeles Crime")
    st.write(" Regression, Classification, Association rule generation")

    # Streamlit app
    st.title("Distribution of Crimes Over Different Years")
    crime['Year'] = pd.to_datetime(crime['DATE OCC']).dt.year
    # Plot the histogram using Plotly Express
    fig1 = px.histogram(crime, x='Year', nbins=14, title="Distribution of Crimes Over Different Years")

    # Customize the layout
    fig1.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        xaxis=dict(tickvals=list(range(2010, 2024))),
        bargap=0.1,
    )

    # Display the plot
    st.plotly_chart(fig1)

    # Create a pie chart using Plotly Express
    fig2= px.pie(crime['AREA NAME'].value_counts(), 
                names=crime['AREA NAME'].value_counts().index,
                values=crime['AREA NAME'].value_counts().values,
                title="Distribution of Crimes in Different Areas",
                hole=0.3,  # Set to 0 for a traditional pie chart
                color_discrete_sequence=px.colors.qualitative.Set3
                )

    # Customize the layout
    fig2.update_layout(
        legend=dict(title="Area Name"),
    )

    # Display the pie chart
    st.plotly_chart(fig2)



def page2():
    st.title("classification")
    # Load preprocessed data and transformer
    X_train, X_valid, X_test, y_train, y_valid, y_test, le_crime, transformer = preprocess_data()
    transformer.fit(X_train)
    # Dropdown to select the classification model
    model_option = st.selectbox("Select Model", ["Random Forest", "SVM", "KNN"])
    # Get user input for hyperparameter tuning
    tune_hyperparameters = st.checkbox("Enable Hyperparameter Tuning", value=False)
    # Train and evaluate the selected model
    # Train and evaluate the selected model
    if model_option == "Random Forest":
        st.subheader("Random Forest Model")
        rf_model = train_random_forest(tune_hyperparameters=tune_hyperparameters)
        if tune_hyperparameters:
            y_test_pred_rf = rf_model.best_estimator_.predict(X_test)
        else:
            # y_test_pred_rf = rf_model.predict(X_test)
            # Load the model
            loaded_rf_model = joblib.load('random_forest_best_model.joblib')

            # Make predictions with the loaded model
            y_test_pred_rf = loaded_rf_model.predict(X_test)
        # y_test_pred_rf = rf_model.best_estimator_.predict(X_test)
        # st.write("Random Forest Test Classification Report:\n", classification_report(y_test, y_test_pred_rf))
        plot_confusion_matrix(y_test, y_test_pred_rf, "Random Forest")

    elif model_option == "SVM":
        st.subheader("SVM Model")
        svm_model = train_svm(tune_hyperparameters=tune_hyperparameters)
        if tune_hyperparameters:
            y_test_pred_svm = svm_model.best_estimator_.predict(X_test)
        else:
            # y_test_pred_rf = rf_model.predict(X_test)
            # Load the model
            loaded_svm_model = joblib.load('svm_best_model.joblib')
            y_test_pred_rf = loaded_svm_model.predict(X_test)
        # st.write("SVM Test Classification Report:\n", classification_report(y_test, y_test_pred_svm))
        plot_confusion_matrix(y_test, y_test_pred_svm, "SVM")

    elif model_option == "KNN":
        st.subheader("KNN Model")
        knn_model = train_knn(tune_hyperparameters=tune_hyperparameters)
        if tune_hyperparameters:
            y_test_pred_knn = knn_model.best_estimator_.predict(X_test)
        else:
            # y_test_pred_rf = rf_model.predict(X_test)
            # Load the model
            loaded_knn_model = joblib.load('knn_best_model.joblib')
            y_test_pred_rf = loaded_knn_model.predict(X_test) 
        # st.write("KNN Test Classification Report:\n", classification_report(y_test, y_test_pred_knn))
        plot_confusion_matrix(y_test, y_test_pred_knn, "KNN")
    # Sidebar for user input
    st.sidebar.header("User Input")

    # Define default values or placeholders for user input
    default_values = {
        'LAT': 0.0,
        'LON': 0.0,
        'Area ID': 0,
        'Time Category': 'morning',
        'Year': 2023,  # Set to the current year or a default value
        'Month': 1,    # Set to a default month
        'Day': 1,      # Set to a default day
        'Weekday': 'Monday',  # Set to a default weekday
        'Is Weekend': 0  # Set to a default value
    }
    # Collect user input for each feature
    user_input = {}
    for feature, default_value in default_values.items():
        if feature == 'Area ID':
            user_input[feature] = st.sidebar.slider(f"Select {feature}", 0, 21, default_value)
        elif feature in ['Time Category', 'Weekday']:
            if feature == 'Time Category':
                user_input[feature] = st.sidebar.selectbox(f"Select {feature}", ['morning', 'afternoon', 'evening', 'midnight'])
            else:
                weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_encoder = LabelEncoder()
                weekday_encoder.fit(weekdays)
                user_input[feature] = weekday_encoder.transform([st.sidebar.selectbox(f"Select {feature}", weekdays)])[0]
                # user_input[feature] = st.sidebar.selectbox(f"Select {feature}", ['morning', 'afternoon', 'evening', 'midnight']) if feature == 'Time Category' else st.sidebar.selectbox(f"Select {feature}", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        elif feature in ['LAT', 'LON']:
            user_input[feature] = st.sidebar.slider(f"Select {feature}", min_value=-90.0, max_value=90.0, value=default_value) if feature == 'LAT' else st.sidebar.slider(f"Select {feature}", min_value=-180.0, max_value=180.0, value=default_value)
        else:
            user_input[feature] = st.sidebar.text_input(f"Enter value for {feature}", default_value)

    # Convert the user input to a DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Preprocess the user input data using the existing transformer
    # user_input_processed = preprocess_user_input(user_input_df, transformer)
    user_input_processed = transformer.transform(user_input_df)

    # Make predictions based on the trained model and user input
    if model_option == "Random Forest":
        st.subheader("Random Forest Prediction")
        prediction = rf_model.best_estimator_.predict(user_input_processed)
    elif model_option == "SVM":
        st.subheader("SVM Prediction")
        prediction = svm_model.best_estimator_.predict(user_input_processed)
    elif model_option == "KNN":
        st.subheader("KNN Prediction")
        prediction = knn_model.best_estimator_.predict(user_input_processed)

    # Display the prediction
    st.write("Model Prediction:", prediction)


def page3():
    st.title("Regression")


def page4():
    st.title("Association rules")

page_names_to_funcs = {
    "EDA": page1,
    "Classification": page2,
    "Regression": page3,
    "Association_Rules": page4
                       }

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
