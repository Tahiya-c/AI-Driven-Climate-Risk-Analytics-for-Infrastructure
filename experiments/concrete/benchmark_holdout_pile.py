# ARCHIVED exploration (Aisha). Superseded by src/models/concrete_benchmark_all.py,
# which runs these models on the shared spec (real data, annual_rate target, trimmed
# features, 2026-forward autoregressive forecast). Kept only for reference. The
# synthetic-data generator that used to live here has been removed -- it now reads
# the real consolidated CSV, but the target/features/split below are the OLD setup.

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler

cwd = Path.cwd()
project_root = cwd if (cwd / "data").exists() else cwd.parents[1]
csv_path = project_root / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"

# ---------------------------------------------------------
# 2. PREPROCESSING & HOLDOUT SETUP
# ---------------------------------------------------------
df = pd.read_csv(csv_path)
target_col = "carb_depth_increment_mm"

drop_cols = [
    "source_row",
    "description",
    "protection",
    "rebar_diameter_mm",
    "carb_depth_xc_mm",
    "carb_initiation_prob",
    "carb_icorr_ua_cm2",
    "chloride_c_at_xc_kgm3",
    "chloride_c_at_cover_kgm3",
    "chloride_initiation_prob",
    "chloride_icorr_ua_cm2",
    "combined_initiation_prob",
    target_col,
]

cat_features = [
    "zone",
    "exposure_class",
    "carbonation_class",
    "chloride_class",
    "emissions_pathway",
    "climate_model",
    "scenario_label",
]

feature_cols = [
    col for col in df.columns if col not in drop_cols and col != "asset_code"
]

for col in cat_features:
    df[col] = df[col].astype(str)

test_asset = "PileA"
train_df = df[df["asset_code"] != test_asset].copy()
test_df = df[df["asset_code"] == test_asset].copy()

# Encodings for standard scikit-learn models
df_encoded = pd.get_dummies(df[feature_cols], columns=cat_features)
scaler = MinMaxScaler()
X_encoded_scaled = scaler.fit_transform(df_encoded)

X_train_enc = X_encoded_scaled[df["asset_code"] != test_asset]
y_train = df[df["asset_code"] != test_asset][target_col]
X_test_enc = X_encoded_scaled[df["asset_code"] == test_asset]
y_test = df[df["asset_code"] == test_asset][target_col]

# ---------------------------------------------------------
# 3. BASELINE MODELS (RIDGE, RANDOM FOREST, CATBOOST)
# ---------------------------------------------------------
print(f"Evaluating models on holdout asset ({test_asset})...")

# Model 1: Ridge Baseline
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_enc, y_train)
ridge_preds = ridge.predict(X_test_enc)

# Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_enc, y_train)
rf_preds = rf.predict(X_test_enc)

# Model 3: CatBoost Regressor
X_train_cb = train_df[feature_cols]
y_train_cb = train_df[target_col]
X_test_cb = test_df[feature_cols]
y_test_cb = test_df[target_col]

train_pool = Pool(X_train_cb, y_train_cb, cat_features=cat_features)
test_pool = Pool(X_test_cb, y_test_cb, cat_features=cat_features)

cb_model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.04,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)
cb_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30)
cb_preds = cb_model.predict(test_pool)


# ---------------------------------------------------------
# 4. PURE NUMPY LSTM REGRESSOR
# ---------------------------------------------------------
class NumpyLSTM:

    def __init__(self, input_dim, hidden_dim=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W = (
            np.random.randn(input_dim, 4 * hidden_dim)
            * np.sqrt(2.0 / (input_dim + hidden_dim))
        )
        self.U = (
            np.random.randn(hidden_dim, 4 * hidden_dim)
            * np.sqrt(2.0 / (2 * hidden_dim))
        )
        self.b = np.zeros((1, 4 * hidden_dim))
        self.b[0, :hidden_dim] = 1.0

        self.W_out = (
            np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / (hidden_dim + 1))
        )
        self.b_out = np.zeros((1, 1))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def forward(self, X_seq):
        batch_size, timesteps, _ = X_seq.shape
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))

        for t in range(timesteps):
            xt = X_seq[:, t, :]
            gates = xt @ self.W + h @ self.U + self.b
            f = self._sigmoid(gates[:, : self.hidden_dim])
            i = self._sigmoid(
                gates[:, self.hidden_dim : 2 * self.hidden_dim]
            )
            c_tilde = np.tanh(
                gates[:, 2 * self.hidden_dim : 3 * self.hidden_dim]
            )
            o = self._sigmoid(gates[:, 3 * self.hidden_dim :])

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

        y_pred = h @ self.W_out + self.b_out
        return y_pred.flatten()

    def fit(self, X_train, y_train, epochs=25, lr=0.01):
        for _ in range(epochs):
            preds = self.forward(X_train)
            grad_out = (preds - y_train) / len(y_train)
            self.W_out -= lr * np.clip(
                np.dot(
                    np.tanh(X_train[:, -1, : self.hidden_dim]).T,
                    grad_out.reshape(-1, 1),
                ),
                -1.0,
                1.0,
            )
            self.b_out -= lr * np.sum(grad_out)


# 10-Year Sliding Sequences
WINDOW_SIZE = 10
df_lstm = df.copy().sort_values(by=["asset_code", "scenario_label", "year"])
scaler_y = MinMaxScaler()
y_scaled_all = scaler_y.fit_transform(df_lstm[[target_col]]).flatten()

X_seq, y_seq, assets = [], [], []
for asset_name in df_lstm["asset_code"].unique():
    for scn in df_lstm["scenario_label"].unique():
        sub_idx = (df_lstm["asset_code"] == asset_name) & (
            df_lstm["scenario_label"] == scn
        )
        sub_X = X_encoded_scaled[sub_idx.values]
        sub_y = y_scaled_all[sub_idx.values]

        for i in range(len(sub_X) - WINDOW_SIZE):
            X_seq.append(sub_X[i : i + WINDOW_SIZE])
            y_seq.append(sub_y[i + WINDOW_SIZE])
            assets.append(asset_name)

X_seq, y_seq, assets = np.array(X_seq), np.array(y_seq), np.array(assets)
train_mask = assets != test_asset
test_mask = assets == test_asset

lstm_model = NumpyLSTM(input_dim=X_seq.shape[2], hidden_dim=16)
lstm_model.fit(X_seq[train_mask], y_seq[train_mask], epochs=30, lr=0.01)

lstm_preds_scaled = lstm_model.forward(X_seq[test_mask])
lstm_preds = scaler_y.inverse_transform(
    lstm_preds_scaled.reshape(-1, 1)
).flatten()
y_test_lstm = scaler_y.inverse_transform(
    y_seq[test_mask].reshape(-1, 1)
).flatten()

# ---------------------------------------------------------
# 5. BENCHMARK SUMMARY
# ---------------------------------------------------------
all_models = {
    "Ridge Regression (Baseline)": (y_test, ridge_preds),
    "Random Forest Regressor": (y_test, rf_preds),
    "CatBoost Regressor": (y_test, cb_preds),
    "LSTM Regressor (Temporal)": (y_test_lstm, lstm_preds),
}

summary_records = []
for model_name, (y_true, y_pred) in all_models.items():
    summary_records.append(
        {
            "Model": model_name,
            "Holdout Asset": test_asset,
            "R2 Score": round(r2_score(y_true, y_pred), 4),
            "RMSE (mm/yr)": round(root_mean_squared_error(y_true, y_pred), 4),
            "MAE (mm/yr)": round(mean_absolute_error(y_true, y_pred), 4),
        }
    )

results_table = pd.DataFrame(summary_records)
print("\n" + "=" * 68)
print("         CONCRETE DETERIORATION MODEL BENCHMARK RESULTS")
print("=" * 68)
print(results_table.to_string(index=False))
print("=" * 68)

# ---------------------------------------------------------
# 6. SAVE RESULTS FOR DASHBOARD
# ---------------------------------------------------------

artifacts_dir = project_root / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

results_path = artifacts_dir / "concrete_model_benchmark.csv"
results_table.to_csv(results_path, index=False)

print(f"\nBenchmark results saved to: {results_path}")

# Save CatBoost model for future dashboard inference
model_path = artifacts_dir / "catboost_concrete.cbm"
cb_model.save_model(str(model_path))

print(f"CatBoost model saved to: {model_path}")
