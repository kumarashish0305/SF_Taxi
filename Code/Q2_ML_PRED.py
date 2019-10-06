import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

#Read the parquet dataframe and convert to pandas dataframe
df = sqlContext.read.parquet('/mnt/parquet2/sf_taxi_new')
df = df.toPandas()

df = df.iloc[:,[1,3,7,8,9]]
df.head(10)

X = df.drop("Waiting_Distance", axis=1)
y = df["Waiting_Distance"]
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Fit a Linear Regression model to compute waiting distance by TaxiID before getting next passenger
# Similarly, Linear Regression model to compute waiting Time by TaxiID can be built before getting next passenger
from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier

classifier.fit(X_train, y_train)

print("Training Data Score: " + str({classifier.score(X_train, y_train)}))
print("Testing Data Score: " + str({classifier.score(X_test, y_test)}))

#Neural network based predictor also built using tensorflow / keras, single neuron and multilayers 
# NN getting trained on the databricks cluster using SGD optimiser
# Waiting distance forms the loss function
