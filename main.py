from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, OPTIONS, etc.)
    allow_headers=["*"],
)

model = joblib.load("news_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


class TextInput(BaseModel):
    text: str
    description: str    

@app.post("/predict")
def predict(data: TextInput):
    X = tfidf.transform([data.text + data.description])
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
