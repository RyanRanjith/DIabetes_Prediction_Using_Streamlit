import numpy as np
import pickle

# Load the scaler and the model together
with open('trained_model.sav', 'rb') as model_file:
    scaler, loaded_model = pickle.load(model_file)

# Your new input data
input_data = (5,166,72,19,175,25.8,0.587,51)

# Convert input data to numpy array and reshape for a single prediction
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Apply the same scaling transformation used during training
std_data = scaler.transform(input_data_as_numpy_array)

# Make the prediction using the loaded model
prediction = loaded_model.predict(std_data)

# Output the result
if prediction[0] == 0:
    print('not diabetic')
else:
    print('diabetic')