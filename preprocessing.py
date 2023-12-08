import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Get balanced data using clean_data function from data_cleaning.py
    data_balance = pd.read_csv("cleaned_data.csv")
    # Copy the data to avoid modifying the original
    data = data_balance.copy()

    # Convert 'DATE OCC' to datetime and extract features
    data['DATE OCC'] = pd.to_datetime(data['DATE OCC'])
    data['Year'] = data['DATE OCC'].dt.year
    data['Month'] = data['DATE OCC'].dt.month
    data['Day'] = data['DATE OCC'].dt.day
    data['Weekday'] = data['DATE OCC'].dt.weekday

    # Convert 'Day Type' to binary (1 for Weekend, 0 for Weekday)
    data['Is Weekend'] = data['Day Type'].apply(lambda x: 1 if x == 'Weekend' else 0)

    # Encode 'Crime Category' with LabelEncoder
    le_crime = LabelEncoder()
    data['Crime Category'] = le_crime.fit_transform(data['Crime Category'])

    # Define categorical and numerical features
    categorical_features = ['Time Category', 'Crime Code Description']
    numerical_features = ['TIME OCC','LAT', 'LON', 'Area ID', 'Year', 'Month', 'Day', 'Weekday', 'Is Weekend']

    # Create transformers for preprocessing
    transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    # Define feature matrix and target vector
    X = data.drop(['Crime Category', 'DATE OCC', 'Day Type','LOCATION'], axis=1)
    y = data['Crime Category']

    # Split data into training and remaining data
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.3, random_state=42)

    # Split the remaining data equally into validation and test sets
    X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, le_crime, transformer,data_balance
