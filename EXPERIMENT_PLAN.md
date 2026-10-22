# Validation search plan

Fixed before the first model search:

- Seed: 42. Split: 876 training / 292 validation / 292 held-out test.
- Both families receive the same 312 transformed features.
- Selection: lowest validation log-RMSE; first configuration wins exact ties.
- Random forest: 200 trees; maximum depth in [8, 16, unlimited]; minimum leaf size in [1, 3]. Six configurations.
- PyTorch MLP: two ReLU hidden layers, width in [64, 128]; learning rate in [0.0003, 0.001, 0.003]. Six configurations.
- Network constants: Adam, batch size 64, weight decay 0.001, MSE on training-centered log prices, at most 400 epochs, early-stopping patience 35, restore lowest-validation-MSE weights.
- Equal configuration counts; record actual training times to describe effort accurately.
- No held-out test evaluation during search. Final evaluation follows completion of search and ablation.

These are development files. The final single-command workflow and submission package are still to be completed.
