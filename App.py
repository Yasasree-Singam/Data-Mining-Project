# import numpy as np
import streamlit as st
# import pandas as pd
import plotly.express as px
# import geopandas as gpd
# import folium
# from folium.plugins import MarkerCluster
# from streamlit_folium import folium_static
# from data_cleaning import clean_data
# from classification import train_random_forest, train_svm, train_knn, plot_confusion_matrix
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# from preprocessing import preprocess_data
# import joblib
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from datetime import datetime
# import holidays
# import sklearn



# # Load and clean data
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



# def page2():
#     st.title("classification")
#     # Load preprocessed data and transformer
#     X_train, X_valid, X_test, y_train, y_valid, y_test, le_crime, transformer = preprocess_data()
#     transformer.fit(X_train)
#     # Dropdown to select the classification model
#     model_option = st.selectbox("Select Model", ["Random Forest", "SVM", "KNN"])
#     # Get user input for hyperparameter tuning
#     tune_hyperparameters = st.checkbox("Enable Hyperparameter Tuning", value=False)
#     if model_option == "Random Forest":
#         st.subheader("Random Forest Model")
#         rf_model = train_random_forest(tune_hyperparameters=tune_hyperparameters)
#         if tune_hyperparameters:
#             y_test_pred_rf = rf_model.best_estimator_.predict(X_test)
#         else:
#             loaded_rf_model = joblib.load('random_forest_best_model.joblib')
#             y_test_pred_rf = loaded_rf_model.predict(X_test)
#             test_accuracy = accuracy_score(y_test, y_test_pred_rf)
#         plot_confusion_matrix(y_test, y_test_pred_rf, "Random Forest")

#     elif model_option == "SVM":
#         st.subheader("SVM Model")
#         svm_model = train_svm(tune_hyperparameters=tune_hyperparameters)
#         if tune_hyperparameters:
#             y_test_pred_svm = svm_model.best_estimator_.predict(X_test)
#         else:
#             loaded_svm_model = joblib.load('svm_best_model.joblib')
#             y_test_pred_svm = loaded_svm_model.predict(X_test)
#             test_accuracy = accuracy_score(y_test, y_test_pred_svm)
#         plot_confusion_matrix(y_test, y_test_pred_svm, "SVM")

#     elif model_option == "KNN":
#         st.subheader("KNN Model")
#         knn_model = train_knn(tune_hyperparameters=tune_hyperparameters)
#         if tune_hyperparameters:
#             y_test_pred_knn = knn_model.best_estimator_.predict(X_test)
#         else:
#             loaded_knn_model = joblib.load('knn_best_model.joblib')
#             y_test_pred_knn = loaded_knn_model.predict(X_test) 
#             test_accuracy = accuracy_score(y_test, y_test_pred_knn)
#         plot_confusion_matrix(y_test, y_test_pred_knn, "KNN")
#     # Sidebar for user input
#     if st.sidebar.button('User Input'):
#         st.sidebar.header("User Input, Select the below options")
#         # Group by 'AREA NAME' and get the minimum and maximum values for 'LAT' and 'LON'
#         area_lat_lon = crime[['AREA NAME', 'LAT', 'LON']].drop_duplicates()
#         area_lat_lon_dict = area_lat_lon.groupby('AREA NAME').agg({'LAT': ['min', 'max'], 'LON': ['min', 'max']}).reset_index()
#         area_lat_lon_dict.columns = ['AREA NAME', 'LAT_min', 'LAT_max', 'LON_min', 'LON_max']
#         selected_area = st.sidebar.selectbox("Select Area", area_lat_lon['AREA NAME'].unique())
#         # Get the corresponding lat, lon values for the selected area
#         area_lat_lon_row = area_lat_lon_dict[area_lat_lon_dict['AREA NAME'] == selected_area]
#         selected_lat_lon = {
#             'LAT': [area_lat_lon_row['LAT_min'].values[0], area_lat_lon_row['LAT_max'].values[0]],
#             'LON': [area_lat_lon_row['LON_min'].values[0], area_lat_lon_row['LON_max'].values[0]],
#         }
#         area_mapping = {
#         'Newton': 13.0, 'Pacific': 14.0, 'Hollywood': 6.0, 'Central': 1.0, 'Northeast': 11.0,
#         'Hollenbeck': 4.0, 'Southwest': 3.0, 'Rampart': 2.0, 'Devonshire': 17.0, 'Southeast': 18.0,
#         'Olympic': 20.0, 'Harbor': 5.0, 'Wilshire': 7.0, '77th Street': 12.0, 'West LA': 8.0,
#         'Topanga': 21.0, 'Mission': 19.0, 'Foothill': 16.0, 'Van Nuys': 9.0, 'N Hollywood': 15.0,
#         'West Valley': 10.0
#         }
#         # Ensure values are in native Python float format and not NaN
#         lat_min = float(area_lat_lon_row['LAT_min'].values[0]) if not pd.isna(area_lat_lon_row['LAT_min'].values[0]) else 0.0
#         lat_max = float(area_lat_lon_row['LAT_max'].values[0]) if not pd.isna(area_lat_lon_row['LAT_max'].values[0]) else 0.0
#         lon_min = float(area_lat_lon_row['LON_min'].values[0]) if not pd.isna(area_lat_lon_row['LON_min'].values[0]) else 0.0
#         lon_max = float(area_lat_lon_row['LON_max'].values[0]) if not pd.isna(area_lat_lon_row['LON_max'].values[0]) else 0.0

#         # Slider for LAT and LON
#         user_input = {
#             'LAT': st.sidebar.slider(
#                 "Select LAT",
#                 min_value=lat_min,
#                 max_value=lat_max,
#                 value=(lat_min + lat_max) / 2
#             ),
#             'LON': st.sidebar.slider(
#                 "Select LON",
#                 min_value=lon_min,
#                 max_value=lon_max,
#                 value=(lon_min + lon_max) / 2
#             ),
#         }
#                 # Convert 'Area Name' to 'Area ID'
#         user_input['Area ID'] = area_mapping[selected_area]


#         # Collecting and parsing date and time input
#         date_input = st.sidebar.text_input("Enter the date (mm/dd/yyyy)", "01/01/2023")
#         time_occ_input = st.sidebar.text_input("Enter the time occurred (hhmm)", "0000")

#         try:
#             selected_date = datetime.strptime(date_input, "%m/%d/%Y")
#             us_holidays = holidays.UnitedStates()
#             user_input['Is Holiday'] = selected_date in us_holidays
#             user_input['DATE_OCC'] = selected_date.strftime("%Y-%m-%d")
#             user_input['Year'] = selected_date.year
#             user_input['Month'] = selected_date.month
#             user_input['Day'] = selected_date.day
#             day_of_week = selected_date.weekday()
#             user_input['Weekday'] = day_of_week
#             user_input['Is Weekend'] = 1 if day_of_week >= 5 else 0

#             # Time category logic
#             time_occ = int(time_occ_input)
#             if 400 <= time_occ < 1200:
#                 user_input['Time Category'] = 'Morning'
#             elif 1200 <= time_occ < 1700:
#                 user_input['Time Category'] = 'Afternoon'
#             elif 1700 <= time_occ < 2200:
#                 user_input['Time Category'] = 'Evening'
#             else:
#                 user_input['Time Category'] = 'Midnight'

#             user_input['TIME OCC'] = time_occ

#         except ValueError:
#             st.sidebar.warning("Invalid date or time format. Please enter the date in mm/dd/yyyy format and time in hhmm format.")

#         user_input['Crime Code Description'] = st.sidebar.text_input("Enter Crime Code Description", "No Traffic Collision")

#         user_input_df = pd.DataFrame([user_input])[X_train.columns]
#         if st.sidebar.button('Start Prediction'):
#             try:
#                 # Make prediction based on the model selected by the user
#                 if model_option == "Random Forest":
#                     st.subheader("Random Forest Prediction")
#                     prediction = loaded_rf_model.predict(user_input_df)
#                 elif model_option == "SVM":
#                     st.subheader("SVM Prediction")
#                     prediction = loaded_svm_model.predict(user_input_df)
#                 elif model_option == "KNN":
#                     st.subheader("KNN Prediction")
#                     prediction = loaded_knn_model.predict(user_input_df)

#                 # Map the numerical prediction to a label based on your mapping
#                 prediction_label = 'No Crime' if prediction[0] == 1 else 'Crime'

#                 # Display the prediction and a corresponding message
#                 st.write(f"Accuracy: {test_accuracy:.2%}")
#                 st.write(f"Prediction: {prediction_label}")
#                 if prediction_label == 'Crime':
#                     st.write("Be careful around this area, there might be criminal activity.")
#                 else:
#                     st.write("It's safe to go out in this area.")
                    
#             except Exception as e:
#                 st.error("An error occurred during prediction. Check your input and try again.")
#                 st.write(e)





#     # # Make predictions based on the trained model and user input
#     # if model_option == "Random Forest":
#     #     st.subheader("Random Forest Prediction")
#     #     try:
#     #         # scaled_features_processed = transformer.transform(user_input_df)
#     #         # Proceed with the prediction
#     #         prediction = loaded_rf_model.predict(processed_user_input)
#     #         st.write("Prediction:", prediction)
#     #     except ValueError as e:
#     #         st.sidebar.error("Error in transforming user input. Please check the input values.")
#     #         print("Error in transforming user input:", e)
#     #     except Exception as e:
#     #         st.sidebar.error("Error in making a prediction.")
#     #         print("Error in prediction:", e)
#     #     # prediction = loaded_rf_model.predict(scaled_features_processed)
#     # elif model_option == "SVM":
#     #     st.subheader("SVM Prediction")
#     #     prediction = svm_model.best_estimator_.predict(processed_user_input)
#     # elif model_option == "KNN":
#     #     st.subheader("KNN Prediction")
#     #     prediction = knn_model.best_estimator_.predict(processed_user_input)

#     # Display the prediction
#     # st.write("Model Prediction:", prediction)


# def page3():
#     st.title("Regression")


# def page4():
#     st.title("Association rules")

page_names_to_funcs = {
    "EDA": page1,
    # "Classification": page2,
    # "Regression": page3,
    # "Association_Rules": page4
                       }

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
