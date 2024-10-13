import numpy as np
import pickle
import streamlit as st

# Load the saved model and scaler
with open('trained_model.sav', 'rb') as model_file:
    scaler, loaded_model = pickle.load(model_file)

# Function for diabetes prediction
def diabetes_prediction(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data_reshaped)

    # Make the prediction using the loaded model
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return 'not diabetic'
    else:
        return 'diabetic'

# Main function for the Streamlit app
def main():
    # Title of the web app
    st.title('Diabetes Prediction Web App')

    # Getting user input for the prediction
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')

    # Placeholder for diagnosis result
    diagnosis = ''

    # Button to trigger prediction
    if st.button('Diabetes Test Result'):
        # Convert user input to list of float values
        input_data = [float(Pregnancies), float(Glucose), float(BloodPressure),
                      float(SkinThickness), float(Insulin), float(BMI),
                      float(DiabetesPedigreeFunction), float(Age)]

        # Call the prediction function
        diagnosis = diabetes_prediction(input_data)

    # Display the result
    st.success(diagnosis)

# Run the app
if __name__ == '__main__':
    main()