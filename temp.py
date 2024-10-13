import numpy as np
import pickle

input_data =(1,103,30,38,83,43.3,0.183,33)

#Changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)


prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0]==0):
  print('not diabetic')
else:
  print('diabetic')