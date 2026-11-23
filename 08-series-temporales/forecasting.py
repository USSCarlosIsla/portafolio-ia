"""Modelos predictivos para series temporales: pronóstico de ventas diarias a 28 días.

Serie sintética de 4 años con lo que tiene una serie real de ventas: tendencia, estacionalidad
semanal y anual, picos en festivos, promociones (conocidas de antemano) y ruido.

Se comparan, con backtesting walk-forward (3 orígenes de pronóstico):
  * Baselines: naive (último valor) y naive estacional (misma semana anterior).
  * Ridge con features de lags y calendario.
  * Gradient Boosting (HistGradientBoosting) con las mismas features + intervalos por cuantiles.
  * LSTM en PyTorch que predice los 28 días de una vez (multi-output).
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from common.utils import output_dir, save_fig

HORIZON = 28  # días a pronosticar
rng = np.random.default_rng(7)
torch.manual_seed(0)


# ======================= 1. Datos =======================
def make_series():
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    t = np.arange(len(dates))
    trend = 200 + 0.08 * t
    weekly = np.array([0.85, 0.9, 0.95, 1.0, 1.15, 1.35, 1.1])[dates.day_of_week]  # sábado pico
    yearly = 1 + 0.15 * np.sin(2 * np.pi * (dates.day_of_year - 80) / 365.25)
    holiday = ((dates.month == 12) & (dates.day.isin([23, 24, 31]))) | ((dates.month == 1) & (dates.day == 1))
    promo = rng.random(len(dates)) < 0.05  # 5% de días con promoción (planificada → conocida a futuro)
    y = trend * weekly * yearly * np.where(holiday, 1.6, 1) * np.where(promo, 1.3, 1)
    y = y * rng.normal(1, 0.06, len(dates))
    return pd.DataFrame({"y": y, "holiday": holiday.astype(int), "promo": promo.astype(int)}, index=dates)


# ======================= 2. Features =======================
def make_features(df):
    """Solo usamos lags >= HORIZON: así, para cualquier día del horizonte, el valor ya era conocido
    en el momento de pronosticar. Esto evita la fuga de información (data leakage) temporal."""
    f = pd.DataFrame(index=df.index)
    for lag in [28, 35, 42, 56, 364]:
        f[f"lag_{lag}"] = df["y"].shift(lag)
    base = df["y"].shift(HORIZON)
    f["roll_mean_7"] = base.rolling(7).mean()
    f["roll_mean_28"] = base.rolling(28).mean()
    f["roll_std_28"] = base.rolling(28).std()
    f["dow"] = df.index.day_of_week
    f["month"] = df.index.month
    f["doy_sin"] = np.sin(2 * np.pi * df.index.day_of_year / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * df.index.day_of_year / 365.25)
    f["t"] = np.arange(len(df))
    f[["holiday", "promo"]] = df[["holiday", "promo"]]  # variables exógenas conocidas a futuro
    f = pd.concat([f, pd.get_dummies(f["dow"], prefix="dow", dtype=float)], axis=1)
    return f


# ======================= 3. Métricas =======================
def metrics(y_true, y_pred, y_train):
    mae = np.mean(np.abs(y_true - y_pred))
    # MASE: MAE relativo al naive estacional (semanal) en entrenamiento. < 1 = mejor que el baseline
    scale = np.mean(np.abs(y_train[7:] - y_train[:-7]))
    return {
        "MAE": mae,
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "MAPE %": 100 * np.mean(np.abs((y_true - y_pred) / y_true)),
        "MASE": mae / scale,
    }


# ======================= 4. Modelos =======================
def forecast_naive(train, n):
    return np.full(n, train[-1])


def forecast_seasonal_naive(train, n, season=7):
    return np.array([train[-season + (i % season)] for i in range(n)])


def forecast_tabular(model, X, y, train_end, test_idx):
    model.fit(X.iloc[:train_end], y[:train_end])
    return model.predict(X.iloc[test_idx])


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # último estado oculto → 28 valores futuros


def forecast_lstm(y_train, lookback=112, epochs=60):
    """Ventanas deslizantes; cada ventana se normaliza por su propia media/escala para que la red
    aprenda la FORMA del patrón y no el nivel absoluto (las LSTM extrapolan mal tendencias)."""
    windows = np.lib.stride_tricks.sliding_window_view(y_train, lookback + HORIZON)
    mu = windows[:, :lookback].mean(1, keepdims=True)
    W = windows / mu
    X = torch.tensor(W[:, :lookback, None], dtype=torch.float32)
    Y = torch.tensor(W[:, lookback:], dtype=torch.float32)
    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            loss = nn.functional.l1_loss(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    last = y_train[-lookback:]
    with torch.no_grad():
        pred = model(torch.tensor(last / last.mean(), dtype=torch.float32)[None, :, None])
    return pred.numpy()[0] * last.mean()


# ======================= 5. Backtesting walk-forward =======================
def main():
    df = make_series()
    X = make_features(df).iloc[364 + HORIZON:]  # descartamos el inicio sin lags completos
    y_all = df["y"].iloc[364 + HORIZON:].to_numpy()
    dates = X.index
    origins = [len(X) - 3 * HORIZON, len(X) - 2 * HORIZON, len(X) - HORIZON]

    # min_samples_leaf=5 (default 20): solo hay ~12 festivos en el histórico; con hojas de 20 muestras
    # el árbol NO puede aprender su efecto. Revisa siempre los defaults frente al tamaño de tus eventos raros.
    gbm = lambda **kw: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=5,
                                                     random_state=0, **kw)
    models = {
        "Naive": lambda tr, te: forecast_naive(y_all[:tr], len(te)),
        "Naive estacional": lambda tr, te: forecast_seasonal_naive(y_all[:tr], len(te)),
        "Ridge + lags": lambda tr, te: forecast_tabular(make_pipeline(StandardScaler(), Ridge(1.0)), X, y_all, tr, te),
        "Gradient Boosting": lambda tr, te: forecast_tabular(gbm(), X, y_all, tr, te),
        "LSTM": lambda tr, te: forecast_lstm(y_all[:tr]),
    }

    scores, last_preds = {name: [] for name in models}, {}
    for k, origin in enumerate(origins, 1):
        test_idx = np.arange(origin, origin + HORIZON)
        print(f"Fold {k}: entrena hasta {dates[origin - 1].date()}, pronostica "
              f"{dates[test_idx[0]].date()} → {dates[test_idx[-1]].date()}")
        for name, fn in models.items():
            pred = fn(origin, test_idx)
            scores[name].append(metrics(y_all[test_idx], pred, y_all[:origin]))
            last_preds[name] = pred

    summary = pd.DataFrame({n: pd.DataFrame(s).mean() for n, s in scores.items()}).T.sort_values("MASE")
    print("\nPromedio de los 3 folds:")
    print(summary.round(3).to_string())
    summary.to_csv(output_dir(__file__) / "backtesting.csv")

    # Intervalos de predicción 80% con regresión cuantílica (último fold)
    tr, te = origins[-1], np.arange(origins[-1], origins[-1] + HORIZON)
    lo = forecast_tabular(gbm(loss="quantile", quantile=0.1), X, y_all, tr, te)
    hi = forecast_tabular(gbm(loss="quantile", quantile=0.9), X, y_all, tr, te)
    coverage = np.mean((y_all[te] >= lo) & (y_all[te] <= hi))
    print(f"\nCobertura del intervalo 80% (último fold): {coverage:.0%}")

    # Importancia de features por permutación (qué está usando el modelo)
    from sklearn.inspection import permutation_importance
    model = gbm().fit(X.iloc[:tr], y_all[:tr])
    imp = permutation_importance(model, X.iloc[te], y_all[te], n_repeats=10, random_state=0)
    top = pd.Series(imp.importances_mean, index=X.columns).sort_values().tail(8)

    fig, ax = plt.subplots(1, 3, figsize=(20, 4.8), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    ax[0].plot(df.index, df["y"], lw=0.6)
    ax[0].set_title("Serie completa: tendencia + estacionalidades + festivos")
    hist = slice(tr - 70, tr)
    ax[1].plot(dates[hist], y_all[hist], color="k", lw=1, label="histórico")
    ax[1].plot(dates[te], y_all[te], color="k", lw=2, label="real")
    for name in ["Naive estacional", "Gradient Boosting", "Ridge + lags"]:
        ax[1].plot(dates[te], last_preds[name], lw=1.5, label=name)
    ax[1].fill_between(dates[te], lo, hi, alpha=0.2, label="intervalo 80% (GBM)")
    ax[1].set_title("Pronóstico a 28 días (último fold)")
    ax[1].legend(fontsize=8)
    ax[1].tick_params(axis="x", rotation=30)
    ax[2].barh(top.index, top.values)
    ax[2].set_title("Importancia por permutación (GBM)")
    save_fig(fig, __file__, "forecasting.png")


if __name__ == "__main__":
    main()
