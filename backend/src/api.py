from fastapi import FastAPI

app = FastAPI(title="Premier League Prediction API")

@app.get("/")
def read_root():
    return {"message": "API is running!"}
