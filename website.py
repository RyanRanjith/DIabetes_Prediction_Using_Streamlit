import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('/Users/ranjithsreekaranuradhagopinath/Desktop/Learning/Machine_learning/Diabetes_Prediction/trained_model.sav','rb'))

# creating a function for prediction

def diabetes_prediction(input_data):
    
    

    #Changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
      return 'not diabetic'
    else:  
       return 'diabetic'
   
   
   
def main():
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    # getting input data from the user
     
    
     
     Pregnancies = st.text_input('Number of Pregnancies')
     Glucose = st.text_input('Glucose Level')
     BloodPressure = st.text_input('Blood Pressure Value')
     SkinThickness = st.text_input('Skin Thickness')
     Insulin = st.text_input('Insulin Level')
     BMI = st.text_input('BMI Value')
     DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
     Age = st.text_input('Age')
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction 
    
    if st.button('Diabetes Test Result'):
      diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if__name__ == '__main__':
   main()