
app = FastAPI()
app.state.model = #MODEL

@app.get("/predict")
# Let's use our model
app.state.model.predict()








@app.get("/")
def root():
    return dict(greeting="Hello")
