import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from classification import train_random_forest, train_svm, train_knn, plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from Apriori_algorithm.apriori import runApriori, dataFromFile, to_str_results
from Data_preprocessing.preprocessing import preprocess_data
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from datetime import datetime
import holidays
import sklearn
import uuid
import time

# Initialize session state variables
if 'user_input_data' not in st.session_state:
    st.session_state['user_input_data'] = None
if 'predict_button_pressed' not in st.session_state:
    st.session_state['predict_button_pressed'] = False

# # Define a function to convert the binary prediction to the required format
# def convert_traffic_prediction(prediction):
#     label_mapping = {0: 'No Traffic Collision', 1: 'TRAFFIC COLLISION'}
#     return label_mapping[prediction]

def collect_user_input(data_balance,X_train):
    st.sidebar.header("User Input, Select the below options")
    # Group by 'AREA NAME' and get the minimum and maximum values for 'LAT' and 'LON'
    area_lat_lon = data_balance[['Area ID', 'LAT', 'LON']].drop_duplicates()
    area_lat_lon_dict = area_lat_lon.groupby('Area ID').agg({'LAT': ['min', 'max'], 'LON': ['min', 'max']}).reset_index()
    area_lat_lon_dict.columns = ['Area ID', 'LAT_min', 'LAT_max', 'LON_min', 'LON_max']
    # selected_area = st.sidebar.selectbox("Select Area", area_lat_lon['Area ID'].unique())
    area_mapping = {
        'Newton': 13.0, 'Pacific': 14.0, 'Hollywood': 6.0, 'Central': 1.0, 'Northeast': 11.0,
        'Hollenbeck': 4.0, 'Southwest': 3.0, 'Rampart': 2.0, 'Devonshire': 17.0, 'Southeast': 18.0,
        'Olympic': 20.0, 'Harbor': 5.0, 'Wilshire': 7.0, '77th Street': 12.0, 'West LA': 8.0,
        'Topanga': 21.0, 'Mission': 19.0, 'Foothill': 16.0, 'Van Nuys': 9.0, 'N Hollywood': 15.0,
        'West Valley': 10.0
        }
    # unique_key = uuid.uuid4()
    # Generate a unique key using the current time
    unique_key = str(time.time()).replace('.', '')
    selected_area = st.sidebar.selectbox("Select Area", list(area_mapping.keys()),key=f'area_select_{unique_key}')
    selected_area_id = area_mapping[selected_area]

    # Get the corresponding lat, lon values for the selected area
    area_lat_lon_row = area_lat_lon_dict[area_lat_lon_dict['Area ID'] == selected_area_id]
    selected_lat_lon = {
            'LAT': [area_lat_lon_row['LAT_min'].values[0], area_lat_lon_row['LAT_max'].values[0]],
            'LON': [area_lat_lon_row['LON_min'].values[0], area_lat_lon_row['LON_max'].values[0]],
        }
    # Ensure values are in native Python float format and not NaN
    lat_min = float(area_lat_lon_row['LAT_min'].values[0]) if not pd.isna(area_lat_lon_row['LAT_min'].values[0]) else 0.0
    lat_max = float(area_lat_lon_row['LAT_max'].values[0]) if not pd.isna(area_lat_lon_row['LAT_max'].values[0]) else 0.0
    lon_min = float(area_lat_lon_row['LON_min'].values[0]) if not pd.isna(area_lat_lon_row['LON_min'].values[0]) else 0.0
    lon_max = float(area_lat_lon_row['LON_max'].values[0]) if not pd.isna(area_lat_lon_row['LON_max'].values[0]) else 0.0

    # Slider for LAT and LON
    user_input = {
            'LAT': st.sidebar.slider(
                "Select LAT",
                min_value=lat_min,
                max_value=lat_max,
                value=(lat_min + lat_max) / 2,
                key=f'lat_slider_{unique_key}'
            ),
            'LON': st.sidebar.slider(
                "Select LON",
                min_value=lon_min,
                max_value=lon_max,
                value=(lon_min + lon_max) / 2,
                key=f'lon_slider_{unique_key}'
            ),
        }
                # Convert 'Area Name' to 'Area ID'
    user_input['Area ID'] = selected_area_id


    # Collecting and parsing date and time input
    date_input = st.sidebar.text_input("Enter the date (mm/dd/yyyy)", "01/01/2023")
    time_occ_input = st.sidebar.text_input("Enter the time occurred (hhmm)", "0000")

    try:
        selected_date = datetime.strptime(date_input, "%m/%d/%Y")
        us_holidays = holidays.UnitedStates()
        user_input['Is Holiday'] = selected_date in us_holidays
        user_input['DATE_OCC'] = selected_date.strftime("%Y-%m-%d")
        user_input['Year'] = selected_date.year
        user_input['Month'] = selected_date.month
        user_input['Day'] = selected_date.day
        day_of_week = selected_date.weekday()
        user_input['Weekday'] = day_of_week
        user_input['Is Weekend'] = 1 if day_of_week >= 5 else 0

        # Time category logic
        time_occ = int(time_occ_input)
        if 400 <= time_occ < 1200:
            user_input['Time Category'] = 'Morning'
        elif 1200 <= time_occ < 1700:
            user_input['Time Category'] = 'Afternoon'
        elif 1700 <= time_occ < 2200:
            user_input['Time Category'] = 'Evening'
        else:
            user_input['Time Category'] = 'Midnight'

        user_input['TIME OCC'] = time_occ

    except ValueError:
        st.sidebar.warning("Invalid date or time format. Please enter the date in mm/dd/yyyy format and time in hhmm format.")
    traffic_model = joblib.load('saved_models/svm_traffic_pipeline.joblib')

    # Prepare the data for traffic model prediction (exclude 'Crime Code Description')
    traffic_model_input = pd.DataFrame([user_input])[['TIME OCC', 'Area ID', 'LAT', 'LON', 'Time Category', 
                                                      'Is Holiday', 'DATE_OCC', 'Year', 'Month', 
                                                      'Day', 'Weekday', 'Is Weekend']]
    # st.write(traffic_model_input)

    # Predict traffic collision (or Crime Code Description) based on the user input
    traffic_collision_prediction = traffic_model.predict(traffic_model_input)

    # Convert the prediction to 'No Traffic Collision' or 'TRAFFIC COLLISION'
    converted_prediction = traffic_collision_prediction[0]

    # Add the converted 'Crime Code Description' to the user input
    user_input['Crime Code Description'] = converted_prediction

    # Create the final DataFrame to be used for crime prediction
    user_input_df = pd.DataFrame([user_input])[X_train.columns]
    
    # st.write("Current user input:", user_input_df)
    return user_input_df


def page1():
    st.title("Los Angeles Crime")
    st.write(" Regression, Classification, Association rule generation")

    # Streamlit app
    # st.title("Distribution of Crimes Over Different Years")
    # crime['Year'] = pd.to_datetime(crime['DATE OCC']).dt.year
    # # Plot the histogram using Plotly Express
    # fig1 = px.histogram(crime, x='Year', nbins=14, title="Distribution of Crimes Over Different Years")

    # # Customize the layout
    # fig1.update_layout(
    #     xaxis_title="Year",
    #     yaxis_title="Number of Crimes",
    #     xaxis=dict(tickvals=list(range(2010, 2024))),
    #     bargap=0.1,
    # )

    # # Display the plot
    # st.plotly_chart(fig1)

    # # Create a pie chart using Plotly Express
    # fig2= px.pie(crime['AREA NAME'].value_counts(), 
    #             names=crime['AREA NAME'].value_counts().index,
    #             values=crime['AREA NAME'].value_counts().values,
    #             title="Distribution of Crimes in Different Areas",
    #             hole=0.3,  # Set to 0 for a traditional pie chart
    #             color_discrete_sequence=px.colors.qualitative.Set3
    #             )

    # # Customize the layout
    # fig2.update_layout(
    #     legend=dict(title="Area Name"),
    # )

    # # Display the pie chart
    # st.plotly_chart(fig2)



def page2():
    st.title("classification")
    # Load preprocessed data and transformer
    X_train, X_valid, X_test, y_train, y_valid, y_test, le_crime, transformer,data_balance = preprocess_data()
    transformer.fit(X_train)
    # Dropdown to select the classification model
    model_option = st.selectbox("Select Model", ["Random Forest", "SVM", "KNN"])
    # Get user input for hyperparameter tuning
    # tune_hyperparameters = st.checkbox("Enable Hyperparameter Tuning", value=False)
    if model_option == "Random Forest":
        st.subheader("Random Forest Model")
        # rf_model = train_random_forest(tune_hyperparameters=tune_hyperparameters)
        # if tune_hyperparameters:
        #     y_test_pred_rf = rf_model.best_estimator_.predict(X_test)
        loaded_rf_model = joblib.load('saved_models/random_forest_best_model.joblib')
        y_test_pred_rf = loaded_rf_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred_rf)
        plot_confusion_matrix(y_test, y_test_pred_rf, "Random Forest")

    elif model_option == "SVM":
        st.subheader("SVM Model")
        # svm_model = train_svm(tune_hyperparameters=tune_hyperparameters)
        # if tune_hyperparameters:
        #     y_test_pred_svm = svm_model.best_estimator_.predict(X_test)
        # else:
        loaded_svm_model = joblib.load('saved_models/svm_best_model.joblib')
        y_test_pred_svm = loaded_svm_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred_svm)
        plot_confusion_matrix(y_test, y_test_pred_svm, "SVM")

    elif model_option == "KNN":
        st.subheader("KNN Model")
        # knn_model = train_knn(tune_hyperparameters=tune_hyperparameters)
        # if tune_hyperparameters:
        #     y_test_pred_knn = knn_model.best_estimator_.predict(X_test)
        # else:
        loaded_knn_model = joblib.load('saved_models/knn_best_model.joblib')
        y_test_pred_knn = loaded_knn_model.predict(X_test) 
        test_accuracy = accuracy_score(y_test, y_test_pred_knn)
        plot_confusion_matrix(y_test, y_test_pred_knn, "KNN")

        
    # Sidebar for user input
    if st.sidebar.button('User Input'):
        st.session_state['user_input_data'] = collect_user_input(data_balance, X_train)

   # Display user input fields and Start Prediction button
    if st.session_state['user_input_data'] is not None:
        st.session_state['user_input_data'] = collect_user_input(data_balance, X_train)
        st.write("User input for prediction:", st.session_state['user_input_data'])

        if st.sidebar.button('Start Prediction'):
            st.session_state['predict_button_pressed'] = True

            
    # Prediction and results display
    if st.session_state.predict_button_pressed:
        try:
            # Ensure that the models are loaded just once
            if 'loaded_rf_model' not in st.session_state:
                st.session_state.loaded_rf_model = joblib.load('saved_models/random_forest_best_model.joblib')
            if 'loaded_svm_model' not in st.session_state:
                st.session_state.loaded_svm_model = joblib.load('saved_models/svm_best_model.joblib')
            if 'loaded_knn_model' not in st.session_state:
                st.session_state.loaded_knn_model = joblib.load('saved_models/knn_best_model.joblib')

            # Perform prediction using the loaded model and user input
            if model_option == "Random Forest":
                st.subheader("Random Forest Prediction")
                # prediction = st.session_state.loaded_rf_model.predict(st.session_state.user_input_data)
                prediction = st.session_state['loaded_rf_model'].predict(st.session_state['user_input_data'])
            elif model_option == "SVM":
                st.subheader("SVM Prediction")
                # prediction = st.session_state.loaded_svm_model.predict(st.session_state.user_input_data)
                prediction = st.session_state['loaded_svm_model'].predict(st.session_state['user_input_data'])
            elif model_option == "KNN":
                st.subheader("KNN Prediction")
                # prediction = st.session_state.loaded_knn_model.predict(st.session_state.user_input_data)
                prediction = st.session_state['loaded_knn_model'].predict(st.session_state['user_input_data'])

                
        # Prediction and results display
    # if st.session_state.predict_button_pressed:
    #     with st.spinner('Predicting...'):    
    #         if model_option == "Random Forest":
    #             st.subheader("Random Forest Prediction")
    #             prediction = loaded_rf_model.predict(user_input_df)
    #             test_accuracy = accuracy_score(y_test, y_test_pred_rf)
    #         elif model_option == "SVM":
    #             st.subheader("SVM Prediction")
    #             prediction = loaded_svm_model.predict(user_input_df)
    #         elif model_option == "KNN":
    #             st.subheader("KNN Prediction")
    #             prediction = loaded_knn_model.predict(user_input_df)
    
    
            # Map the numerical prediction to a label based on your mapping
            prediction_label = 'No Crime' if prediction[0] == 1 else 'Crime'
            st.success('Prediction complete!')
            st.write(f"Prediction: {prediction_label}")
            st.write(f"Accuracy: {test_accuracy:.2%}")
            if prediction_label == 'Crime':
                st.write("Be careful around this area, there might be criminal activity.")
            else:
                st.write("It's safe to go out in this area.")
            st.session_state['predict_button_pressed'] = False
        
        except Exception as e:
            st.error("An error occurred during prediction. Check your input and try again.")
            st.write(e)    
                        
                # except Exception as e:
                #     st.error("An error occurred during prediction. Check your input and try again.")
                #     st.write(e)




def page3():
    st.title("Regression")


def page4():
    st.title("Association rules")
    # csv_file = pd.read_csv('combinedapriori.csv')
    st.markdown('''
            **Support** shows transactions with items purchased together in a single transaction.
            
            **Confidence** shows transactions where the items are purchased one after the other.''')

    st.markdown('Support and Confidence for Itemsets A and B can be represented by formulas')

    support_helper = ''' > Support(A) = (Number of transactions in which A appears)/(Total Number of Transactions') '''
    confidence_helper = ''' > Confidence(A->B) = Support(AUB)/Support(A)') '''
    st.markdown('---')

    support = st.slider("Enter the Minimum Support Value", min_value=0.1,
                    max_value=0.9, value=0.15,
                    help=support_helper)

    confidence = st.slider("Enter the Minimum Confidence Value", min_value=0.1,
                       max_value=0.9, value=0.6, help=confidence_helper)

    # Example usage in a Streamlit app
    # inFile = st.file_uploader('Upload File', type=['csv'])
    
    # if inFile is not None:
    #     for record in dataFromFile(inFile):
    #         # Process each record as needed
    #         st.write(record)

    inFile = dataFromFile('combinedapriori.csv')

    items, rules = runApriori(inFile, support, confidence)

    i, r, original_rules = to_str_results(items, rules)

    st.markdown("## Results")

    st.markdown("### Frequent Itemsets")
    st.write(i)

    st.markdown("### Frequent Rules")
    st.write(original_rules)

    st.markdown("### Interpreted Rules")
    st.write(r)

    st.markdown("### Temporal Analysis")

    # Set a confidence threshold
    confidence_threshold = 0.7

    # Perform iterative temporal analysis
    temporal_periods = ['Morning', 'Afternoon', 'Evening', 'Night']

    for temporal_period in temporal_periods:
        print(f"\nTemporal Analysis for {temporal_period}:")
        
        # Filter rules based on confidence threshold and temporal period
        filtered_rules = [
            rule for rule in original_rules
            if (
                len(rule.split(" , ")) > 1
                and float(rule.split(" , ")[1]) >= confidence_threshold
                and temporal_period in rule
            )
]


        # Interpretation of filtered rules
        for rule_str in filtered_rules:
            # Extract information from the rule string
            rule_parts = rule_str.split(" ==> ")
            antecedent_str = rule_parts[0].replace("Rule: (", "").replace(")", "")
            consequent_str = rule_parts[1].split(" , ")[0].replace("(", "").replace(")", "")
            confidence = float(rule_parts[1].split(" , ")[1])

            # Convert antecedents and consequents to lists
            antecedents = [item.strip("'") for item in antecedent_str.split(", ")]
            consequents = [item.strip("'") for item in consequent_str.split(", ")]

            # Interpretation
            antecedent_description = ', '.join(antecedents)
            consequent_description = ', '.join(consequents)
            st.write(f"If {antecedent_description} occurs, then {consequent_description} is likely to occur with confidence {confidence:.3f}")

page_names_to_funcs = {
    "EDA": page1,
    "Classification": page2,
    "Regression": page3,
    "Association_Rules": page4
                       }

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
