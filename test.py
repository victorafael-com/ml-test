import numpy as np
import pandas as pd
import keras
from keras.models import load_model

model = load_model("model_trained.h5")


data = pd.read_csv('data/test.csv') #No need to change =P

X = dataset.iloc[:,:20].values
Y = dataset.iloc[:,20:21].values
