import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import keras
import random
from keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model("model_trained.h5")

data = pd.read_csv('data/test.csv') #No need to change =P

X = data.iloc[:,:20].values
Y = data.iloc[:,20:21].values

labels = ["$30 - $100", "$100 - $210", "$210 - $300", "$300+"]

#fits data
sc = StandardScaler()
X = sc.fit_transform(X)

#ohe = OneHotEncoder()
#Y = ohe.fit_transform(Y).toarray()

line = random.randrange(0, len(X))

print("testing line ", line + 1, "/", len(X))
x_val = X[line]
y_val = Y[line]

print("Expected result on this line: ", labels[y_val[0]])

res = model.predict(X)

loaded = list()
values = list()
for i in range(len(res)):
    loaded.append(np.argmax(Y[i]))
    values.append(np.argmax(res[i]))

a = accuracy_score(loaded, values)

print("Got: ", labels[ values[line] ])
print("overall accuracy: ", a * 100, "%")
