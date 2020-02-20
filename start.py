#Made following:
# https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5

#Dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#Dataset Import
dataset = pd.read_csv('data/train.csv') #No need to change =P
testdata = pd.read_csv('data/test.csv')

#print(dataset.head(10))

X = dataset.iloc[:,:20].values
Y = dataset.iloc[:,20:21].values

X_t = dataset.iloc[:,:20].values
Y_t = dataset.iloc[:,20:21].values

#Standard Scaler
#fits the data between -1 and 1

sc = StandardScaler()
X = sc.fit_transform(X)
X_t = sc.fit_transform(X_t)
from sklearn.model_selection import train_test_split

#One hot encoder:
#converts from
# 0 => [1, 0, 0]
# 2 => [0, 0, 1]
# 1 => [0, 1, 0]

ohe = OneHotEncoder()
Y = ohe.fit_transform(Y).toarray()
Y_t = ohe.fit_transform(Y_t).toarray()


#X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1)

#Create neural network
#Sequential means each layer output is used by input from the next layer
#Dense means that all outputs are connected to all inputs

model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#trains
#trains without validating with test data 
#history = model.fit(X_train, Y_train, epochs=100, batch_size=64)
#tranins validating with test data
history = model.fit(X, Y, validation_data = (X_t, Y_t), epochs=100, batch_size=64)

#test the training
Y_pred = model.predict(X_t)
pred = list()
test = list()
for i in range(len(Y_pred)):
    pred.append(np.argmax(Y_pred[i]))
    test.append(np.argmax(Y_t[i]))

a = accuracy_score(pred, test)
print('Accuracy result: ', a*100,'%')

#save model

model.save('model_trained.h5')

#plot result

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Test', 'Train'], loc='upper left')
plt.show()