import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

PROC_DIR = "examen_dvc/data/processed"
MODELS_DIR = "examen_dvc/models/models"
os.makedirs(MODELS_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(PROC_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join("examen_dvc/data/processed", "y_train.csv")).squeeze()

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_grid = {
"n_estimators": [100, 300, 500],
"max_depth": [None, 5, 10, 20],
"min_samples_split": [2, 5, 10],
"min_samples_leaf": [1, 2, 4]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(rf, param_grid=param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
gs.fit(X_train, y_train)

best_params = gs.best_params_

# Sauvegarde des meilleurs params
import joblib
joblib.dump(best_params, os.path.join(MODELS_DIR, "best_params.pkl"))

# (optionnel) garder un json lisible
with open(os.path.join(MODELS_DIR, "best_params.json"), "w") as f:
 json.dump(best_params, f, indent=2)

print("GridSearch OK.")