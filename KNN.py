import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump, load

# Replace this with the actual file path on your system
file = 'diabetes_prediction_dataset.csv'

# Load data from the hardcoded file path
data = pd.read_csv(file)

# Handle missing values (if any)
imputer = SimpleImputer(strategy='mean')  # You can change strategy as needed

# Assuming the columns 'gender' and 'smoking_history' are categorical
label_encoders = {}

# Encode categorical columns
for col in ['gender', 'smoking_history']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the encoder if needed later for inverse transform

# Handle missing values for numerical columns
data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = imputer.fit_transform(
    data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
)

# Split data into features (X) and target variable (y)
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Save the trained model for later use
dump(knn, 'KNN.joblib')
dump(scaler, 'scaler.joblib')  # Save the scaler as well

# Load the trained model and scaler
loaded_model = load('KNN.joblib')
loaded_scaler = load('scaler.joblib')

# --- Streamlit UI ---
st.title('Diabetes Prediction App')

# Ensure that selectbox options match the ones used during training
gender_options = label_encoders['gender'].classes_.tolist()
smoking_history_options = label_encoders['smoking_history'].classes_.tolist()

# Create UI elements for user input
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', gender_options)
hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
smoking_history = st.selectbox('Smoking History', smoking_history_options)
bmi = st.number_input('BMI', min_value=15, max_value=60)
hb_a1c_level = st.number_input('HbA1c Level', min_value=3, max_value=15)
blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, max_value=500)

# Button to trigger prediction
if st.button('Predict'):
    # Convert categorical data from user input into the same format used during training
    gender_encoded = label_encoders['gender'].transform([gender])[0]
    smoking_history_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]

    # Ensure the new data has the same columns as the training data
    expected_columns = X.columns

    # Create a DataFrame with the same columns as X_train
    new_data = pd.DataFrame({
        'age': [age], 
        'gender': [gender_encoded],  
        'hypertension': [1 if hypertension == 'Yes' else 0],
        'heart_disease': [1 if heart_disease == 'Yes' else 0],
        'smoking_history': [smoking_history_encoded],
        'bmi': [bmi],  
        'HbA1c_level': [hb_a1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

    # Add any missing columns with default values if necessary
    for column in expected_columns:
        if column not in new_data.columns:
            new_data[column] = 0

    # Ensure columns are in the same order as in X_train
    new_data = new_data[expected_columns]

    # Apply the same scaling as the training data
    new_data_scaled = loaded_scaler.transform(new_data)

    # Make a prediction
    prediction = loaded_model.predict(new_data_scaled)

    # Display the prediction with informative messages
    if prediction[0] == 1:
        st.error('Predicted: You are at a higher risk of diabetes. Please consult a healthcare professional.')
    else:
        st.success('Predicted: You are at a lower risk of diabetes based on this model. It is still recommended to maintain a healthy lifestyle and consult a healthcare professional for regular checkups.')
