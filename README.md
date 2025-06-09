# (ON HOLD) Gurobi with Parameter Tuning in MIP Problems

## Introduction
This project investigates the use of machine learning to predict the runtime of MIP models solved by Gurobi under different parameter configurations, based on static problem features.

The primary goal is to see if the gurobi default parameter settings are good enough for solving MIP problems, using MIPLIB Benchmark. The ultimate goal is to learn a model that can recommend effective parameter sets for unseen instances, using runtime as the performance target.

## Current Status
-  Data collection pipeline implemented: Gurobi was run across 27 parameter sets on hundreds of MIP instances.
-  Benchmark features extracted from CSV (variables, constraints, nonzeros, etc.)
-  Runtime data aligned and classified into categories: Optimal, Quasi-optimal, Not Converged, No Feasible.
-  Neural network and tree-based regressors (Random Forest, XGBoost) implemented to predict runtime.
-  However, **no reliable predictive model was obtained**, since the error can not down to a reasonable threshold in terms of not overfitting. Also the best model that I currently get always yields a R^2 scores around zero, which means no predictive results got for evaluate dataset.

Therefore, the project is temporarily on hold until a more suitable instance set becomes available and key technical challenges are addressed.

## Current Limitations explained in detail

Despite the MLP model achieving stable convergence during training (with both training and validation L1 losses decreasing smoothly to ~0.77), the test-time generalization performance remains unsatisfactory:

- Test MAE: ~40.7 seconds
- Test R^2: -0.004 (worse than predicting the mean)
- Systematic Underprediction: The model consistently underestimates long-runtime instances (e.g., 90s predicted as ~20–30s).

The training process cost too much epochs even after applying early stopping, and it always stops at val loss around 0.78-0.88, and cannot converge to 0. I've tried to lower or raise the learning rate, increase or decrease the batch size, change mlp layers and neurons, adding dropout or so on, but the result didn't improved significantly. And the best model always has a R^2 score around 0 indicates that the learnbing is failed.

## Possible Issues

- Long-tailed target distribution: Most training instances have small runtimes (e.g., <20s), while high-runtime cases (e.g., 90s) are underrepresented. This causes the model to focus on "easy" regions of the loss and ignore extremes.
- Direct regression may be unsuitable: Predicting absolute runtime is difficult due to the complex nonlinear interaction between parameters and instance features.
- Smooth loss does not imply good decisions: Even when L1 or MSE loss decreases, the model may still fail to correctly rank or identify the best parameter settings.
- Features from benchmark set are limited to structural metrics (e.g., variable counts), with no semantic or instance type information.
- The way to determine param sets can be false. See the appendix.
- Runtime distribution is long-tailed; models struggled to generalize beyond 10–20 seconds scale. This's possibly because I manually set Gurobi solver time limit to 90s in order to shorten the generating time of dataset.
- Around 4500 instance-param pairs available after filtering which might not be enough.

## Future Work

- Exploring alternative parameter types, such as `Heuristics`, `NodeMethod`, or `Threads`, that may be more strongly correlated with runtime performance.
- Using data-driven approaches to reduce the permutation space (e.g., feature-based filtering, sensitivity analysis).
- Expand the dataset by adding more MIP instances from other domains (e.g., scheduling, cutting stock, etc.)
- Incorporate instance-type metadata or clustering results as features.
- Use runtime gaps and objective values as auxiliary prediction targets.
- Explore classification-based recommendation (e.g., "which param_id beats baseline?") instead of regression.

This direction assumes access to a larger, more diverse dataset to support more statistically meaningful conclusions.

## Project Structure
```bash
MIP-PARAMETER-TUNING/
├── data/
│ ├── instances/
│ │ ├── model_instances.txt # List of benchmark MIP instances
│ │ └── starter_instances.txt # Small starter instance subset
│ ├── params/
│ │ └── param_sets.json # 27 predefined Gurobi parameter sets
│ ├── processed/
│ │ ├── modeldataset.csv # Main processed training dataset
│ │ └── starterdataset.csv # Starter dataset(only contain 2 instances)
├── model/ # directory for model outputs
│
├── notebook/
│ ├── 01_data_preparation.ipynb # Preprocessing and instance chunking
│ ├── 02_param_set.ipynb # Parameter set definition and validation
│ ├── 03_dataset_analysis.ipynb # Feature and label analysis
│ ├── 04_train_mlp_model.ipynb # MLP model training and evaluation
│ └── 05_train_tree_model.ipynb # Tree-based model training (RF/XGB)
│
├── src/
│ ├── generate_dataset.py # Run Gurobi and log runtime results
│ ├── model_utils.py # Feature processing, evaluation utilities
│ ├── predict_param.py # Placeholder for param recommendation logic
│ ├── train_model.py # MLP training script (PyTorch)
│ └── tree_models.py # XGBoost / RandomForest training script
│
├── config.yaml # YAML config file (for future extensions)
├── .gitignore 
├── LICENSE 
└── README.md 
```

## Appendix: Parameter Selection Reference
Parameter Guidelines: https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html
Reference: https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html

By referring to this page, I decided to focus on three different params that might affect MIP solving: 
1. MIPFocus:[1,2,3]
2. Cutting Planes:[0,-1,1,2]
3. Presolve:[0,-1,1,2]
The reasons are: they are all mentioned specificly in part of MIP Problems of the page "Parameter Guidelines", and they seems to affect the solving results significantly. Also here I would like to use MIPGap as Termination control. 
The negative results suggest that my initial choice of the three parameters—or the way their combinations were defined—might have been flawed. However, I have not yet identified a more promising alternative. Thus this project now is temporarily on hold.