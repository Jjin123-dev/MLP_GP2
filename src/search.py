"""Validation-only searches. No held-out test inputs are accepted here."""

import copy
import random
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn


SEED = 42
# Fixed in advance: 6 configurations per family, lowest validation log-RMSE wins.
TREE_GRID = [
    {"max_depth": depth, "min_samples_leaf": leaf, "n_estimators": 200}
    for depth in (8, 16, None) for leaf in (1, 3)
]
NETWORK_GRID = [
    {"learning_rate": rate, "hidden_width": width, "batch_size": 64,
     "weight_decay": 0.001, "max_epochs": 400, "patience": 35}
    for rate in (0.0003, 0.001, 0.003) for width in (64, 128)
]


def log_rmse(target, prediction):
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def fit_network(x_train, y_train, x_val, y_val, config):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # Center the target using training statistics to simplify optimization.
    target_mean = float(y_train.mean())
    inputs = torch.as_tensor(x_train, dtype=torch.float32)
    targets = torch.as_tensor(y_train - target_mean, dtype=torch.float32).reshape(-1, 1)
    val_inputs = torch.as_tensor(x_val, dtype=torch.float32)
    val_targets = torch.as_tensor(y_val - target_mean, dtype=torch.float32).reshape(-1, 1)
    width = config["hidden_width"]
    model = nn.Sequential(nn.Linear(x_train.shape[1], width), nn.ReLU(),
                          nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_function = nn.MSELoss()
    best_loss, best_epoch, best_weights = float("inf"), 0, None
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(inputs))
        for batch in order.split(config["batch_size"]):
            optimizer.zero_grad()
            loss = loss_function(model(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(loss_function(model(inputs), targets))
            val_loss = float(loss_function(model(val_inputs), val_targets))
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config["patience"]:
            break
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        predictions = model(val_inputs).numpy().ravel() + target_mean
    return model, target_mean, log_rmse(y_val, predictions), best_epoch, history


def run_search(x_train, y_train, x_val, y_val, root):
    import json
    records = []
    selected_tree, selected_network = None, None
    best_tree, best_network = float("inf"), float("inf")
    for index, config in enumerate(TREE_GRID, 1):
        started = time.perf_counter()
        model = RandomForestRegressor(**config, random_state=SEED, n_jobs=2)
        model.fit(x_train, y_train)
        score = log_rmse(y_val, model.predict(x_val))
        record = {"family": "trees", "config_id": f"T{index}",
                  "hyperparameters": config, "validation_log_rmse": score,
                  "runtime_seconds": time.perf_counter() - started}
        records.append(record)
        print(f"T{index}: validation log-RMSE={score:.6f}", flush=True)
        if score < best_tree:
            best_tree, selected_tree = score, (model, record)
    for index, config in enumerate(NETWORK_GRID, 1):
        started = time.perf_counter()
        model, mean, score, epoch, history = fit_network(x_train, y_train, x_val, y_val, config)
        record = {"family": "network", "config_id": f"N{index}",
                  "hyperparameters": config, "validation_log_rmse": score,
                  "best_epoch": epoch, "epochs_run": len(history),
                  "runtime_seconds": time.perf_counter() - started}
        records.append(record)
        print(f"N{index}: validation log-RMSE={score:.6f}; best epoch={epoch}", flush=True)
        if score < best_network:
            best_network, selected_network = score, (model, mean, record, history)
    result = {"selection_rule": "lowest validation log-RMSE; first config wins exact ties",
              "seed": SEED, "tree_grid": TREE_GRID, "network_grid": NETWORK_GRID,
              "configurations": records, "selected_tree": selected_tree[1],
              "selected_network": selected_network[2], "test_evaluated": False}
    (root / "search_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (root / "selected_network_history.json").write_text(
        json.dumps(selected_network[3], indent=2) + "\n", encoding="utf-8")
    # Intermediate files retain training-part fits for the later final evaluation.
    import joblib
    joblib.dump(selected_tree[0], root / "artifacts" / "selected_tree.joblib")
    torch.save({"state_dict": selected_network[0].state_dict(),
                "target_mean": selected_network[1], "record": selected_network[2],
                "input_features": x_train.shape[1]}, root / "artifacts" / "selected_network.pt")
    return result
