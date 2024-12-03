import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


from arrhythmia.ml_logic.preproc import preproc, df_from_bucket, label_encoding

def initialize_model():
    return Sequential([Input((1, 180)),
                       LSTM(64),
                       Dense(4, activation='sigmoid')])

def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, df_tr):
    X_tr, y_tr = df_tr.drop(columns='target'), df_tr.target
    X_tr = np.expand_dims(X_tr, axis = 1)
    y_tr_ohe = to_categorical(y_tr)
    
    es = EarlyStopping(patience = 10, restore_best_weights=True)
    history = model.fit(X_tr, y_tr_ohe,
                        epochs=100, batch_size=256,
                        validation_split=0.2,
                        callbacks=[es])
    
    return model, history

def save_model(model, path):
    model.save(path)

def evaluate_model(model, df_te):
    X_te, y_te = df_te.drop(columns='target'), df_te.target
    X_te = np.expand_dims(X_te, axis = 1)
    y_te_ohe = to_categorical(y_te)
    y_pred = model.predict(X_te)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_te, y_pred))
    print(confusion_matrix(y_te, y_pred))
    return model.evaluate(X_te, y_te_ohe)



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # df = df_from_bucket()
    df = pd.read_csv('../arrhythmia_raw_data/MIT-BIH_raw.csv')
    df_tr, df_te = preproc(df, drop_classes=['F'])
    df_tr, df_te = label_encoding([df_tr, df_te], '/home/fabricio/lstm_multi_label_encoding.pkl')
    model = initialize_model()
    model = compile_model(model)
    model, history = train_model(model, df_tr)
    save_model(model, '/home/fabricio/lstm_multi_model.h5')
    evaluate_model(model, df_te)
