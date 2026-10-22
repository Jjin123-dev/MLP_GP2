"""Reproduce the shared pipeline, searches, and single final evaluation."""

from pathlib import Path
import json
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import prepare_features, build_preprocessor


SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def split_data(houses):
    """Return the same 60/20/20 split for every future experiment."""
    development, test = train_test_split(
        houses, test_size=0.20, random_state=SEED
    )
    train, validation = train_test_split(
        development, test_size=0.25, random_state=SEED
    )
    assert (len(train), len(validation), len(test)) == (876, 292, 292)
    assert set(train.Id).isdisjoint(validation.Id)
    assert set(train.Id).isdisjoint(test.Id)
    assert set(validation.Id).isdisjoint(test.Id)
    return train, validation, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="Run both validation-only searches")
    parser.add_argument("--finalize", action="store_true", help="Evaluate saved selected models once")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    # Preserve the literal category "None" (e.g. no masonry veneer).
    csv_options = {"keep_default_na": False, "na_values": ["", "NA"]}
    houses = pd.read_csv(ROOT / "data" / "train.csv", **csv_options)
    competition = pd.read_csv(ROOT / "data" / "test.csv", **csv_options)
    train, validation, test = split_data(houses)

    training_features = prepare_features(train)
    validation_features = prepare_features(validation)
    preprocessor = build_preprocessor(training_features)
    x_train = preprocessor.fit_transform(training_features).astype(np.float32)
    x_validation = preprocessor.transform(validation_features).astype(np.float32)
    x_competition = preprocessor.transform(prepare_features(competition)).astype(np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()
    for matrix in (x_train, x_validation, x_competition):
        assert matrix.shape[1] == len(feature_names)
        assert np.isfinite(matrix).all()
    assert len(set(feature_names)) == len(feature_names)
    preprocessing_record = {
        "fitted_on": "training portion only (876 rows)",
        "training_shape": list(x_train.shape),
        "validation_shape": list(x_validation.shape),
        "competition_shape": list(x_competition.shape),
        "feature_names": feature_names,
        "numeric_missing": "training median; all-missing training columns use zero",
        "numeric_scaling": "training mean and standard deviation",
        "categorical_missing": "dictionary-defined absence -> Absent; otherwise Unknown",
        "categorical_encoding": "training-fitted one-hot; unseen levels -> all zeros",
        "MSSubClass": "categorical",
        "literal_None": "preserved as category",
        "test_transformed": False,
    }
    (ROOT / "preprocessing_results.json").write_text(
        json.dumps(preprocessing_record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Shared input shapes: train={x_train.shape}, validation={x_validation.shape}")

    # Learn the constant using only training prices.
    mean_log_price = float(np.log(train["SalePrice"]).mean())
    validation_targets = np.log(validation["SalePrice"].to_numpy())
    validation_rmse = float(
        np.sqrt(np.mean((validation_targets - mean_log_price) ** 2))
    )
    constant_price = float(np.exp(mean_log_price))
    submission = pd.DataFrame({
        "Id": competition["Id"],
        "SalePrice": constant_price,
    })
    assert len(submission) == 1459
    assert submission["Id"].is_unique
    assert np.isfinite(submission["SalePrice"]).all()
    assert (submission["SalePrice"] > 0).all()
    submission.to_csv(ROOT / "submission_baseline.csv", index=False)

    # intermediate record for the baseline stage, before any search or final evaluation.
    results = {
        "stage": "constant_baseline",
        "seed": SEED,
        "split_sizes": {
            "train": len(train), "validation": len(validation), "test": len(test)
        },
        "baseline": {
            "mean_training_log_price": mean_log_price,
            "constant_price": constant_price,
            "validation_log_rmse": validation_rmse,
        },
        "test_evaluated": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (ROOT / "baseline_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print("Split sizes: training=876, validation=292, test=292")
    print(f"Baseline validation log-RMSE: {validation_rmse:.6f}")
    print(f"Constant predicted price: ${constant_price:,.2f}")
    full_run = not (args.search or args.finalize or args.baseline_only)
    if args.search or full_run:
        from search import run_search
        import joblib
        (ROOT / "artifacts").mkdir(exist_ok=True)
        joblib.dump(preprocessor, ROOT / "artifacts" / "preprocessor.joblib")
        run_search(x_train, np.log(train["SalePrice"].to_numpy()),
                   x_validation, validation_targets, ROOT)
        print(f"Total runtime: {time.perf_counter() - started:.1f} seconds", flush=True)
    if args.finalize or full_run:
        from evaluation import finalize
        prior_search_runtime = 0
        if args.finalize:
            saved_search = json.loads((ROOT / "search_results.json").read_text())
            prior_search_runtime = sum(row["runtime_seconds"] for row in saved_search["configurations"])
        finalize(ROOT, test, competition, x_competition, preprocessor,
                 mean_log_price, validation_rmse,
                 time.perf_counter() - started + prior_search_runtime)


if __name__ == "__main__":
    main()
