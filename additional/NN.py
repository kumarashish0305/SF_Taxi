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
