from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI()

# Get absolute path to the parent directory (root of your repo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Full paths to your pickle files
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
SYMPTOM_PATH = os.path.join(BASE_DIR, "symptom_list.pkl")

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

