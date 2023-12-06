from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
import io
import streamlit as st


# Load preprocessed data and transformer
X_train, X_valid, X_test, y_train, y_valid, y_test, le_crime, transformer = preprocess_data()

def train_random_forest(tune_hyperparameters=True, param_grid=None):
    # Default hyperparameters
    default_params = {
        'n_estimators': 300,
        'max_depth': 10,
        'criterion': 'gini',
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42
    }
    # Define a pipeline for Random Forest
    rf_pipeline = Pipeline([
        ('preprocessor', transformer),
        ('classifier', RandomForestClassifier(**default_params) if not tune_hyperparameters else RandomForestClassifier())
    ])
    if tune_hyperparameters:
        # Perform GridSearchCV as before

        # Define a parameter grid for Random Forest
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20],
            # Add other parameters here
        }

        # Create a GridSearchCV object and fit it to the training data
        grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Evaluate on the validation set
        y_valid_pred = grid_search.predict(X_valid)
        print("Random Forest Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
        print("Random Forest Validation Classification Report:\n", classification_report(y_valid, y_valid_pred))
        return grid_search
    else:
        # Train with default hyperparameters
        rf_pipeline.fit(X_train, y_train)
        y_valid_pred = rf_pipeline.predict(X_valid)
        print("Random Forest Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
        return rf_pipeline

def train_svm(tune_hyperparameters=True, param_grid=None):
    # Define a pipeline for SVM
    # Default hyperparameters
    default_params = {
        'C': 1.0,
        'kernel': 'rbf',
        'random_state': 42
    }

    svm_pipeline = Pipeline([
        ('preprocessor', transformer),
        ('classifier', SVC(**default_params) if not tune_hyperparameters else SVC())
    ])
    if tune_hyperparameters:
        # Perform GridSearchCV as before
    # Define a parameter grid for SVM
        param_grid = {
            'classifier__C': [0.1, 1, 10],  # SVM regularization parameter
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient for 'rbf'
            # Add other parameters here if needed
        }

        # Create a GridSearchCV object for SVM
        grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Evaluate on the validation set
        y_valid_pred = grid_search.predict(X_valid)
        print("SVM Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
        print("SVM Validation Classification Report:\n", classification_report(y_valid, y_valid_pred))


        return grid_search
    else:
        # Train with default hyperparameters
        svm_pipeline.fit(X_train, y_train)
        y_valid_pred = svm_pipeline.predict(X_valid)
        print("SVM Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
        return svm_pipeline

def train_knn(tune_hyperparameters=True, param_grid=None):
    # Define a pipeline for KNN
    # Default hyperparameters
    default_params = {
        'n_neighbors': 7,
        'weights': 'uniform',
        'algorithm': 'ball_tree'
    }

    knn_pipeline = Pipeline([
        ('preprocessor', transformer),
        ('classifier', KNeighborsClassifier(**default_params) if not tune_hyperparameters else KNeighborsClassifier())
    ])
    if tune_hyperparameters:
        # Define a parameter grid for KNN
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7],  # Number of neighbors to use
            'classifier__weights': ['uniform', 'distance'],  # Weight function used in prediction
            'classifier__algorithm': ['ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
            # Add other parameters here if needed
        }

        # Create a GridSearchCV object for KNN
        grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Evaluate on the validation set
        y_valid_pred = grid_search.predict(X_valid)
        print("KNN Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
        print("KNN Validation Classification Report:\n", classification_report(y_valid, y_valid_pred))

        return grid_search
    else:
        # Train with default hyperparameters
        knn_pipeline.fit(X_train, y_train)
        y_valid_pred = knn_pipeline.predict(X_valid)
        print("KNN Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
        return knn_pipeline

def plot_confusion_matrix(y_true, y_pred, model_name):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create the ConfusionMatrixDisplay object
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Display the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cmd.plot(ax=ax)
    plt.title(f'Confusion Matrix - {model_name}')

    # Convert the plot to a PNG image
    image = fig_to_image(fig)

    # Display the image in Streamlit
    st.image(image, caption=f'Confusion Matrix - {model_name}', use_column_width=True)


# Helper function to convert Matplotlib figure to PNG image
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf