from fastapi import FastAPI
from arrhythmia.ml_logic.binary.cnn import apply_cnn
from arrhythmia.ml_logic.preproc import preproc

app = FastAPI()

@app.get("/")
def root():
    return dict(greeting="Hello")

# app.state.model.predict()
app.state.model = apply_cnn()


def initialize_model(X_train):
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
    return model

@app.get("/predict")
# Let's use our model
def predict(filename):
    #load and split data
    data=pd.read_csv(filename)

    model = app.state.model
    assert model is not None

    X_processed = preproc(X_pred)
    y_pred = model.predict(X_processed)

    return (wavetype=float(y_pred))
