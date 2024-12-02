from fastapi import FastAPI



app = FastAPI()

@app.get("/")
def root():
    return dict(greeting="Hello")


# app.state.model = #MODEL

# @app.get("/predict")
# # Let's use our model
# app.state.model.predict()
