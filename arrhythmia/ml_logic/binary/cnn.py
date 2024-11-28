import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

def apply_cnn(filename):
    #load and split data
    data = pd.read_csv(filename)
    X,y = data.drop(columns=['Unnamed: 0','target']),data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #adapt dimensions for the Con1D layer
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    #Initialize the model
    model = models.Sequential()

    model.add(layers.Conv1D(8,3, activation='relu', input_shape=X_train.shape[1:]))

    model.add(layers.Conv1D(16,3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='relu'))

    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy','recall','precision'])

    model = initialize_model(X_train)

    #fit the model
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_split=0.2,
                callbacks=[es])

    res = model.evaluate(X_test, y_test, verbose = 1 )
    print("Vos performances :   "+res)
