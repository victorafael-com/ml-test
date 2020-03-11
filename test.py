import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import keras
import random
from keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model("model_trained.h5")

dataTrain = pd.read_csv('data/train.csv') #No need to change =P
X_train = dataTrain.iloc[:,:20].values
#fits data
sc = StandardScaler()
sc.fit(X_train)


data = pd.read_csv('data/test.csv') #No need to change =P

X = data.iloc[:,1:21].values

line = random.randrange(0, len(X))

print("testing line ", line + 1, "/", len(X))

labels = ["$30 - $100", "$100 - $210", "$210 - $300", "$300+"]


X_line = data.iloc[line:line+1,1:21].values

X_fit = sc.transform(X_line)

res = model.predict(X_fit)
#
#loaded = list()
#values = list()
#for i in range(len(res)):
#    loaded.append(Y[i])
#    values.append(np.argmax(res[i]))
#

print("You requested info about this mobile phone:")

heads = data.columns

#for i in range(len(heads) - 1):
#    pre = "#{:02d}:".format(i)
#    print(pre, heads[i+1], ":", X_line[0][i])

print("\n##############################")
print("# PRICE RANGE:", labels[np.argmax(res[0])])
print("##############################\n")

print("enabling bluetooth, 4g, 3g and touch support would be this price:")

enables = [1,5,17,18,19]

for i in range(len(enables)):
    X_line[0][enables[i]] = 1.0

X_line[0][0] = 1700
X_line[0][2] = 2
X_line[0][9] = 6

X_custom = sc.transform(X_line)

# for i in range(len(X_custom[0])):
    # X_custom[0][i] = X_custom[0][1] * -1

res2 = model.predict(X_custom)


#for i in range(len(heads) - 1):
#    pre = "#{:02d}:".format(i)
#    print(pre, heads[i+1], ":", X_line[0][i])

print("\n##############################")
print("# PRICE RANGE:", labels[np.argmax(res2[0])])
print("##############################\n")