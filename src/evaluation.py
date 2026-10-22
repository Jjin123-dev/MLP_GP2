"""Freeze validation comparisons before the single final test evaluation."""

import json
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from preprocessing import prepare_features
from search import log_rmse


def finalize(root, test, competition, x_competition, preprocessor,
             mean_log_price, baseline_validation, runtime_before_final):
    import time
    started = time.perf_counter()
    search = json.loads((root / "search_results.json").read_text())
    configs = {row["config_id"]: row for row in search["configurations"]}
    # Comparisons fixed before inspecting held-out outcomes.
    ablation = [{"row": 1, "configuration": "constant baseline", "changed_from_row": None,
                 "factor": "none", "validation_log_rmse": baseline_validation}]
    for row, config_id, parent, factor in (
        (2, "T1", 1, "predictor: constant to random forest"),
        (3, "T3", 2, "maximum depth: 8 to 16"),
        (4, "T5", 3, "maximum depth: 16 to unlimited"),
        (5, "T6", 4, "minimum leaf size: 1 to 3"),
        (6, "N5", 1, "predictor: constant to neural network"),
        (7, "N6", 6, "hidden width: 64 to 128"),
    ):
        ablation.append({"row": row, "configuration": config_id,
                         "changed_from_row": parent, "factor": factor,
                         "validation_log_rmse": configs[config_id]["validation_log_rmse"]})
    pd.DataFrame(ablation).to_csv(root / "ablation.csv", index=False)
    tree = joblib.load(root / "artifacts" / "selected_tree.joblib")
    checkpoint = torch.load(root / "artifacts" / "selected_network.pt", weights_only=True)
    width = checkpoint["record"]["hyperparameters"]["hidden_width"]
    network = nn.Sequential(nn.Linear(checkpoint["input_features"], width), nn.ReLU(),
                            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    # SINGLE FINAL TEST EVALUATION: no model or hyperparameter selection below.
    x_test = preprocessor.transform(prepare_features(test)).astype(np.float32)
    y_test = np.log(test["SalePrice"].to_numpy())
    assert np.isfinite(x_test).all()
    with torch.no_grad():
        network_test = network(torch.from_numpy(x_test)).numpy().ravel() + checkpoint["target_mean"]
        network_competition = network(torch.from_numpy(x_competition)).numpy().ravel() + checkpoint["target_mean"]
    predictions = {
        "constant_baseline": np.full(len(y_test), mean_log_price),
        "trees": tree.predict(x_test),
        "neural_network": network_test,
    }
    validation_scores = {"constant_baseline": baseline_validation,
                         "trees": search["selected_tree"]["validation_log_rmse"],
                         "neural_network": search["selected_network"]["validation_log_rmse"]}
    table = [{"model": name, "validation_log_rmse": validation_scores[name],
              "test_log_rmse": log_rmse(y_test, prediction)}
             for name, prediction in predictions.items()]
    for family, log_predictions in (("tree", tree.predict(x_competition)),
                                    ("network", network_competition)):
        submission = pd.DataFrame({"Id": competition["Id"], "SalePrice": np.exp(log_predictions)})
        assert len(submission) == 1459 and submission.Id.is_unique
        assert np.isfinite(submission.SalePrice).all() and (submission.SalePrice > 0).all()
        submission.to_csv(root / f"submission_{family}.csv", index=False)
    # Provisional default chosen by validation.
    winner = "network" if validation_scores["neural_network"] < validation_scores["trees"] else "tree"
    shutil.copyfile(root / f"submission_{winner}.csv", root / "submission.csv")
    pd.DataFrame(table).to_csv(root / "final_comparison.csv", index=False)
    results = {"seed": 42, "split_sizes": {"train": 876, "validation": 292, "test": 292},
               "runtime_seconds": runtime_before_final + time.perf_counter() - started,
               "final_models": table, "search": search, "ablation": ablation,
               "test_evaluation_passes": 1,
               "kaggle_fit": "876-row training portion; no refit",
               "submission_csv_selection": f"{winner}: validation winner; public scores pending"}
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for row in table:
        print(f"{row['model']}: validation={row['validation_log_rmse']:.6f}, test={row['test_log_rmse']:.6f}")
