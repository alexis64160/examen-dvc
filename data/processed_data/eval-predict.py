import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PROC_DIR = "examen_dvc/data"
X_test = pd.read_csv(os.path.join(PROC_DIR, "processed", "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(PROC_DIR, "processed", "y_test.csv")).squeeze()

model = joblib.load(os.path.join("examen_dvc/models/models", "model.pkl"))

# Prédictions
y_pred = model.predict(X_test)

# Sauvegarde des prédictions
preds_df = pd.DataFrame({
"y_true": y_test,
"y_pred": y_pred,
"residual": y_test - y_pred
})
os.makedirs(os.path.join("examen_dvc", "data"), exist_ok=True)
preds_df.to_csv(os.path.join("examen_dvc", "data", "predictions.csv"), index=False)

# Métriques
mse = float(mean_squared_error(y_test, y_pred))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

scores = {"mse": mse, "mae": mae, "r2": r2}

# Ecrire metrics/scores.json
METRICS_DIR = os.path.join("examen_dvc", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
with open(os.path.join(METRICS_DIR, "scores.json"), "w") as f:
 json.dump(scores, f, indent=2)

print("Evaluation OK.")