import pickle
import csv
import calendar
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator, FormatStrFormatter
from scipy.interpolate import spline
import tensorflow as tf 
from sklearn.preprocessing import scale 
from keras.models import Sequential 
from keras.optimizers import SGD, Adam, Adagrad 
from keras import backend as K 
from keras.layers.embeddings import Embedding 
from keras.layers.core import Dense, Reshape, Merge, Activation, Dropout 
from keras.callbacks import ModelCheckpoint 


# Gather some metadata that will later be useful during training 
        metadata = { 
             'dayhour': 24,  # Number of hours in one day (i.e. 24). 
             'Dayperweek': 7, 
             'Weekyear': 52,
             'TaxiID': len(TaxiID_ENCODED.classes_)
        } 



def create_model(metadata, clusters):
    """
    Creates all the layers for our neural network model.
    """

    # Arbitrary dimension for all embeddings
    embedding_dim = 10

    # Quarter hour of the day embedding
    embed_dayhour = Sequential()
    embed_dayhour.add(Embedding(metadata['Dayhour'], embedding_dim, input_length=1))
    embed_dayhour.add(Reshape((embedding_dim,)))

    # Day of the week embedding
    embed_Dayofweek = Sequential()
    embed_Dayofweek.add(Embedding(metadata['Dayofweek'], embedding_dim, input_length=1))
    embed_Dayofweek.add(Reshape((embedding_dim,)))

    # Week of the year embedding
    embed_Weekofyear = Sequential()
    embed_Weekofyear.add(Embedding(metadata['Weekofyear'], embedding_dim, input_length=1))
    embed_Weekofyear.add(Reshape((embedding_dim,)))


    # Taxi ID embedding
    embed_taxi_ids = Sequential()
    embed_taxi_ids.add(Embedding(metadata['TaxiID'], embedding_dim, input_length=1))
    embed_taxi_ids.add(Reshape((embedding_dim,)))


    # GPS coordinates
    coords = Sequential()
    coords.add(Embedding(metadata['coordinates'], embedding_dim, input_length=1))
    coords.add(Reshape((embedding_dim,)))


    # Merge all the inputs into a single input layer
    model = Sequential()
    model.add(Merge([
                embed_dayhour,
                embed_Dayofweek,
                embed_Weekofyear,
                embed_taxi_ids,
                coords
            ], mode='concat'))

    # Simple hidden layer
    model.add(Dense(500))
    model.add(Activation('relu'))

    # Determine cluster probabilities using softmax
    model.add(Dense(len(clusters)))
    model.add(Activation('softmax'))

    # Final activation layer: calculate the destination as the weighted mean of cluster coordinates
    cast_clusters = K.cast_to_floatx(clusters)
    def destination(probabilities):
        return tf.matmul(probabilities, cast_clusters)
    model.add(Activation(destination))

    # Compile the model
    optimizer = SGD(lr=0.01, momentum=0.9, clipvalue=1.)  # Use `clipvalue` to prevent exploding gradients
    model.compile(loss=Waiting_Distance, optimizer=optimizer)

    return model

clusters = pd.DataFrame({
    'approx_latitudes': df['latitude'].round(4),
    'approx_longitudes': df['longitude'].round(4)
})

bandwidth = estimate_bandwidth(clusters, quantile=0.0002)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(clusters)
clusters = ms.cluster_centers_

# Estimate clusters from all destination points
clusters = get_clusters(data.train_labels)
print("Number of estimated clusters: %d" % len(clusters))

plt.figure(figsize=(6,6))
plt.scatter(clusters[:,1], clusters[:,0], c='#cccccc', s=2)
plt.axis('off')
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().autoscale_view('tight')

