from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load saved files
model = pickle.load(open("disease_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
symptoms = pickle.load(open("symptom_list.pkl", "rb"))

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

