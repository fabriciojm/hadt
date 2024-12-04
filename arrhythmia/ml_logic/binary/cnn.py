import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import os
from arrhythmia.ml_logic.preproc import preproc
from arrhythmia.ml_logic.preproc import label_encoding


def initialize_model():
    model = models.Sequential()

    # model.add(layers.Conv1D(8,3, activation='relu', input_shape=X_train.shape[1:]))
    model.add(layers.Conv1D(8,3, activation='relu', input_shape=(180, 1)))

    model.add(layers.Conv1D(16,3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization()) #normalise  pour accélerer entrainement

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))


    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy','recall','precision'])
    return model


def apply_cnn(filename):
    #load and split data
    data = pd.read_csv(filename)
    datatrain, datatest =preproc(data, drop_classes=['F'],binary=True)
    datatrain, datatest = label_encoding([datatrain, datatest],'/Users/france/code/fabriciojm/arrhythmia/arrhythmia/ml_logic/pickles/binary_cnn_labels.pkl')

    X_train,y_train = datatrain.drop(columns=['target']), datatrain.target
    X_test,y_test = datatest.drop(columns=['target']), datatest.target

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #adapt dimensions for the Con1D layer
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    #Initialize the model
    model = initialize_model()

    #fit the model
    es = EarlyStopping(patience=5, restore_best_weights=True)
    print('je suis en train de fiter, ça peut prendre 2min')

    model.fit(X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_split=0.2,
                callbacks=[es],
                verbose =1)

    #affiche les resultats
    res = model.evaluate(X_test, y_test, verbose = 1 )
    resultats= f"Le modèle a été entrainé sur les données indiquées : {(res)}."
    print(resultats)
    y_pred = model.predict(X_test)

    y_pred = (y_pred > 0.5).astype(int) # Converti les probas en prédictions binaires (0 ou 1)

    print(classification_report(y_test, y_pred))

# def save_model(model: keras.Model = None) -> None:

#     if MODEL_TARGET == "gcs":
#             # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

#             model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
#             client = storage.Client()
#             bucket = client.bucket(BUCKET_NAME)
#             blob = bucket.blob(f"models/{model_filename}")
#             blob.upload_from_filename(model_path)

#             print("✅ Model saved to GCS")
#             return None

#     if MODEL_TARGET == "mlflow":
#         mlflow.tensorflow.log_model(
#             model=model,
#             artifact_path="model",
#             registered_model_name=MLFLOW_MODEL_NAME
#         print("✅ Model saved to MLflow")
#         return None
#     return None

if __name__=="__main__":
 #   apply_cnn(../../../)
    rootpath=(os.path.dirname(__file__)) #print(os.getcwd())
    relativepath="../../../raw_data/MIT-BIH_binary_4classes.csv"
    csv_path=os.path.join(rootpath,relativepath)
    print(csv_path)
    print(apply_cnn(csv_path))
