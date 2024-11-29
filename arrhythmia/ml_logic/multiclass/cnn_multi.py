import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

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

def apply_cnn(filename):
    #load and split data
    data = pd.read_csv('../raw_data/MIT-BIH_dropF.csv')
    X,y = data.drop(columns=['target']),data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #adapt dimensions for the Con1D layer
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    #remap y labels to be able to encode
    y_train_unique = np.unique(y_train)
    y_test_unique = np.unique(y_test)
    label_mapping = {old_label: i for i, old_label in enumerate(y_train_unique)}
    label_mapping2 = {old_label: i for i, old_label in enumerate(y_test_unique)}
    y_train_remapped = np.vectorize(label_mapping.get)(y_train)
    y_test_remapped = np.vectorize(label_mapping2.get)(y_test)

    #OHE encode
    y_cat_train = to_categorical(y_train_remapped, num_classes=4)
    y_cat_test = to_categorical(y_test_remapped, num_classes=4)

    #Initialize the model
    model = models.Sequential()

    model.add(layers.Conv1D(8,3, activation='relu', input_shape=X_train.shape[1:]))

    model.add(layers.Conv1D(16,3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization()) #normalise  pour accélerer entrainement
    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy','recall','precision'])

    model = initialize_model(X_train)

    #fit the model
    es = EarlyStopping(patience=30, restore_best_weights=True)

    model.fit(X_train, y_cat_train,
            batch_size=32,
            epochs=200,
            validation_split=0.2,
            callbacks=[es])

    res = model.evaluate(X_test, y_test, verbose = 1 )
    print(res)
