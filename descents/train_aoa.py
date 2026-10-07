"""Multiple linear regression: predict angle of attack (aoa_deg) from flight data.

Split is 70/30 by file (14 train / 6 test from 20 total)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import lstsq

TARGET = "aoa_deg"
FEATURES = [
    "altitude_rel_m",
    "airspeed_mps",
    "pitch_deg",
    "descent_rate_mps",
    "servo_throttle_pwm",  
    "pilot_roll_pwm",       # RC commands 
    "pilot_pitch_pwm",
    "pilot_yaw_pwm",
]


# Read every CSV, keep only the needed columns
def load_all(folder):
    frames = []
    for path in sorted(Path(folder).glob("*.csv")):
        d = pd.read_csv(path, usecols=FEATURES + [TARGET]).dropna()
        d["file"] = path.stem
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# Split training and test files
def split_by_file(df, train_frac=0.7, seed=42):
    files = df["file"].unique()
    rng = np.random.default_rng(seed)
    files = rng.permutation(files)
    cut = int(round(train_frac * len(files)))
    train_files, test_files = files[:cut], files[cut:]
    return df[df["file"].isin(train_files)], df[df["file"].isin(test_files)], train_files, test_files


# Standardize on training data, then solve least squares with an intercept
def fit(train):
    X = train[FEATURES]
    keep = [c for c in FEATURES if X[c].std() > 0]   
    dropped = [c for c in FEATURES if c not in keep]
    mu, sd = X[keep].mean(), X[keep].std()
    A = np.column_stack([np.ones(len(train)), ((X[keep] - mu) / sd).to_numpy()])
    coef, *_ = lstsq(A, train[TARGET].to_numpy())
    return {"coef": coef, "keep": keep, "mu": mu, "sd": sd, "dropped": dropped}


def predict(model, df):
    Z = ((df[model["keep"]] - model["mu"]) / model["sd"]).to_numpy()
    return np.column_stack([np.ones(len(df)), Z]) @ model["coef"]


def score(y, pred, baseline):
    resid = y - pred
    return {
        "R2": 1 - np.sum(resid**2) / np.sum((y - y.mean())**2),
        "RMSE": float(np.sqrt(np.mean(resid**2))),
        "baseline RMSE (always predict train mean)": float(np.sqrt(np.mean((y - baseline)**2))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="descents")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = load_all(a.folder)
    train, test, tr_files, te_files = split_by_file(df, seed=a.seed)
    print(f"{df['file'].nunique()} files, {len(df)} rows -> "
          f"train {len(tr_files)} files / {len(train)} rows, test {len(te_files)} files / {len(test)} rows")

    print("Train files:", ", ".join(sorted(tr_files)))
    print("Test files: ", ", ".join(sorted(te_files)))

    model = fit(train)
    if model["dropped"]:
        print("Dropped (constant in training data):", model["dropped"])

    print("\n=== Coefficients (standardized, so sizes are comparable) ===")
    print(f"{'intercept':<22}{model['coef'][0]:>10.4f}")
    for name, c in sorted(zip(model["keep"], model["coef"][1:]), key=lambda t: -abs(t[1])):
        print(f"{name:<22}{c:>10.4f}")

    base = train[TARGET].mean()
    for label, d in [("TRAIN", train), ("TEST", test)]:
        s = score(d[TARGET].to_numpy(), predict(model, d), base)
        print(f"\n=== {label} ===")
        for k, v in s.items():
            print(f"{k:<45}{v:>9.4f}")

    pred = predict(model, test)
    y = test[TARGET].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y, pred, alpha=0.4)
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax[0].plot(lims, lims, color="red", lw=1)
    ax[0].set(xlabel="Actual AOA (deg)", ylabel="Predicted AOA (deg)", title="Test Set")
    ax[1].scatter(pred, y - pred, alpha=0.4)
    ax[1].axhline(0, color="red", lw=1)
    ax[1].set(xlabel="Predicted AOA (deg)", ylabel="Residual (deg)", title="Residuals")
    fig.tight_layout()
    fig.savefig("aoa_results.png", dpi=120)
    print("\nSaved aoa_results.png")


if __name__ == "__main__":
    main()