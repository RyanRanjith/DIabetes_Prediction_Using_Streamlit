import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('/Users/ranjithsreekaranuradhagopinath/Desktop/Learning/Machine_learning/Diabetes_Prediction/trained_model.sav','rb'))

input_data =(5,166,72,19,175,25.8,0.587,51)

#Changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('not diabetic')
else:
  print('diabetic')