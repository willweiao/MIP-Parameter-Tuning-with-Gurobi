# (ON HOLD) Gurobi with Parameter Tuning in MIP Problems

## Introduction
This project investigates the use of machine learning to predict the runtime of MIP models solved by Gurobi under different parameter configurations, based on static problem features.

The primary goal is to see if the gurobi default parameter settings are good enough for solving MIP problems, using MIPLIB Benchmark. The ultimate goal is to learn a model that can recommend effective parameter sets for unseen instances, using runtime as the performance target.

## Current Status
-  Data collection pipeline implemented: Gurobi was run across 27 parameter sets on hundreds of MIP instances.
-  Benchmark features extracted from CSV (variables, constraints, nonzeros, etc.)
-  Runtime data aligned and classified into categories: Optimal, Quasi-optimal, Not Converged, No Feasible.
-  Neural network and tree-based regressors (Random Forest, XGBoost) implemented to predict runtime.
-  However, **no reliable predictive model was obtained**, likely due to the limited size and variance in the dataset.

Thus this project is currently on hold, pending access to a larger and more diverse instance set.

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

## Limitations

- Only ~300–1000 instance-param pairs available after filtering.
- Features from benchmark set are limited to structural metrics (e.g., variable counts), with no semantic or instance type information.
- Runtime distribution is long-tailed; models struggled to generalize beyond 10–20 seconds scale.
- R² scores remain close to zero, indicating poor predictive power on validation/test data.

I suspect the issue lies primarily in dataset sparsity and lack of strong signal in the features.

### Future Work

- Expand the dataset by adding more MIP instances from other domains (e.g., scheduling, cutting stock, etc.)
- Incorporate instance-type metadata or clustering results as features.
- Use runtime gaps and objective values as auxiliary prediction targets.
- Explore classification-based recommendation (e.g., "which param_id beats baseline?") instead of regression.

If a large enough dataset becomes available, I expect this pipeline to scale effectively.

## Parameter Selection Reference
Parameter Guidelines: https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html
Reference: https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html

By referring to this page, I decided to focus on three different params that might affect MIP solving: 
1. MIPFocus:[1,2,3]
2. Cutting Planes:[0,-1,1,2]
3. Presolve:[0,-1,1,2]
The reasons are: they are all mentioned specificly in part of MIP Problems of the page "Parameter Guidelines", and they seems to affect the solving results significantly.
Also here I would like to use MIPGap as Termination control. 