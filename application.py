import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# AWS Elastic Beanstalk requires the app object to be named 'application'
application = Flask(__name__)
app = application  # alias for local dev

MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"
DATA_PATH = "kc_house_data.csv"

FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "zipcode"
]

FEATURE_LABELS = {
    "bedrooms":    ("Bedrooms",    "int",   1, 10,    3),
    "bathrooms":   ("Bathrooms",   "float", 0.5, 8, 2.0),
    "sqft_living": ("Sqft Living", "int",   200, 10000, 1800),
    "zipcode":     ("Zipcode",     "int",   98001, 98199, 98052),
}

model_metrics = {}


def train_and_save():
    """Train the KNN model on the dataset and persist to disk."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["id", "date"], axis=1)

    X = df.drop("price", axis=1)[FEATURES]
    y = df["price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return knn, scaler, {"mse": round(mse, 2), "r2": round(r2, 4), "rmse": round(mse ** 0.5, 2)}


def load_or_train():
    """Load persisted model/scaler, or train fresh if missing."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        knn = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # Recompute quick metrics from a small test run
        df = pd.read_csv(DATA_PATH)
        df = df.drop(["id", "date"], axis=1)
        X = df.drop("price", axis=1)[FEATURES]
        y = df["price"]
        scaler_check = scaler
        X_scaled = scaler_check.transform(X)
        _, X_test, _, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        y_pred = knn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {"mse": round(mse, 2), "r2": round(r2, 4), "rmse": round(mse ** 0.5, 2)}
        return knn, scaler, metrics
    return train_and_save()


# ── Bootstrap model on startup ──────────────────────────────────────────────
knn_model, std_scaler, model_metrics = load_or_train()
print(f"[KNN] Model ready  R²={model_metrics['r2']}  RMSE=${model_metrics['rmse']:,.0f}")


# ── Routes ───────────────────────────────────────────────────────────────────
@application.route("/")
def index():
    return render_template(
        "index.html",
        features=FEATURES,
        feature_labels=FEATURE_LABELS,
        metrics=model_metrics,
    )


@application.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        for feat in FEATURES:
            raw = request.form.get(feat, "0")
            ftype = FEATURE_LABELS[feat][1]
            values.append(float(raw) if ftype == "float" else int(raw))

        arr = np.array(values).reshape(1, -1)
        arr_scaled = std_scaler.transform(arr)
        predicted_price = knn_model.predict(arr_scaled)[0]

        input_data = {FEATURE_LABELS[f][0]: v for f, v in zip(FEATURES, values)}

        return render_template(
            "result.html",
            predicted_price=f"${predicted_price:,.0f}",
            input_data=input_data,
            metrics=model_metrics,
        )
    except Exception as e:
        return render_template("index.html",
                               features=FEATURES,
                               feature_labels=FEATURE_LABELS,
                               metrics=model_metrics,
                               error=str(e)), 400


@application.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint for programmatic access."""
    try:
        data = request.get_json(force=True)
        values = []
        for feat in FEATURES:
            ftype = FEATURE_LABELS[feat][1]
            raw = data.get(feat, FEATURE_LABELS[feat][4])
            values.append(float(raw) if ftype == "float" else int(raw))

        arr = np.array(values).reshape(1, -1)
        arr_scaled = std_scaler.transform(arr)
        predicted_price = float(knn_model.predict(arr_scaled)[0])

        return jsonify({"predicted_price": predicted_price, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400


@application.route("/retrain", methods=["POST"])
def retrain():
    """Retrain the model from the CSV (admin endpoint)."""
    global knn_model, std_scaler, model_metrics
    knn_model, std_scaler, model_metrics = train_and_save()
    return jsonify({"status": "ok", "metrics": model_metrics})


@application.route("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    application.run(debug=False, host="0.0.0.0", port=port)
