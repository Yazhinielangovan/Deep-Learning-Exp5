# Deep-Learning-Exp5

DL-Implement a Recurrent Neural Network model for stock price prediction.

**AIM**

To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

**THEORY**

**Neural Network Model**

<img width="994" height="496" alt="image" src="https://github.com/user-attachments/assets/9ae44eac-8524-479e-805b-6017be39d952" />

**DESIGN STEPS**

Step-1 Read the CSV file and create the DataFrame using pandas.

Step-2 Select the “Open” column (or any desired column) and scale the values using MinMaxScaler.

Step-3 Create two lists — X_train and y_train — where X_train stores 60 readings as input and the 61st reading as output in y_train.

Step-4 Build an LSTM model with the desired number of neurons and a single output neuron.

Step-5 Combine the training and test data, then prepare X_test using the same 60-step sequence logic.

Step-6 Use the trained model to make predictions on the test data and inverse transform the results to their original scale.

Step-7 Plot the graph comparing the Actual and Predicted stock prices using matplotlib.

**PROGRAM**

**Name: Yazhini **

**Register Number: 2305002028 **


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential([layers.SimpleRNN(40,input_shape=(60,1)),
                    layers.Dense(1)])
model.compile(optimizer='adam',loss='mse')
model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(X_train1,y_train,epochs=25, batch_size=64)

model.summary()
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


**OUTPUT**

Epoch Training:

<img width="653" height="416" alt="image" src="https://github.com/user-attachments/assets/1d2c3c66-c023-4ee7-8ef9-12ffd8832578" />

Model Training Loss Across Epochs:

<img width="910" height="579" alt="image" src="https://github.com/user-attachments/assets/75ce684a-b5ec-49b6-b0fe-71f22470ef6a" />


**True Stock Price, Predicted Stock Price vs time**

<img width="721" height="563" alt="image" src="https://github.com/user-attachments/assets/b8512dd4-24d2-4051-bae3-d6c392df3178" />


**Predictions**

<img width="543" height="58" alt="image" src="https://github.com/user-attachments/assets/ecbf578c-37e9-4ed8-b170-cefcd9631688" />

**RESULT**

Thus, a reccurrent neural network for Stock Price Prediction developed successfully.


