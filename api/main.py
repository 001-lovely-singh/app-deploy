from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os
import pathlib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = pathlib.Path(__file__).parent  # points to /api folder on Vercel

MODEL_PATH = BASE_DIR / "disease_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
SYMPTOM_PATH = BASE_DIR / "symptom_list.pkl"


# Load saved files safely
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(SYMPTOM_PATH, "rb") as f:
        symptoms = pickle.load(f)
    print("✅ Model, encoder, and symptom list loaded successfully.")
except Exception as e:
    print("❌ Error loading pickle files:", e)
    model = None
    encoder = None
    symptoms = []

# Root route (for testing)
@app.get("/")
def home():
    return {"message": "FastAPI is running successfully on Vercel!"}

# Input model for POST request
class SymptomInput(BaseModel):
    symptoms: list[str]


@app.post("/predict")
def predict(data: SymptomInput):
    # Clean input (same way as training)
    cleaned_input = [s.strip().lower().replace(" ", "_") for s in data.symptoms]

    # Clean model symptom list (just in case)
    symptoms_cleaned = [s.strip().lower().replace(" ", "_") for s in symptoms]

    input_data = {sym: [1 if sym in cleaned_input else 0] for sym in symptoms_cleaned}
    df = pd.DataFrame(input_data)

    print("👉 Cleaned input symptoms:", cleaned_input)
    print("👉 Binary vector:", df.values.tolist())

    pred = model.predict(df)[0]
    disease = encoder.inverse_transform([pred])[0]
    print("👉 Predicted disease:", disease)
    return {"prediction": disease.replace("_", " ").upper()}

