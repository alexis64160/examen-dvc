import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROC_DIR = "examen_dvc/data/processed"
MODELS_DIR = "examen_dvc/models/models"
os.makedirs(MODELS_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join("examen_dvc/data/processed", "y_train.csv")).squeeze()

best_params = joblib.load(os.path.join(MODELS_DIR, "best_params.pkl"))
model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
model.fit(X_train, y_train)

joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
print("Entraînement OK.")