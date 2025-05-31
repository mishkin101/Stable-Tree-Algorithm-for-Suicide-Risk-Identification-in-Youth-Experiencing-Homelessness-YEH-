# Stable Decision Tree Method for Predicting Suicidal Ideation for At-Risk Homeless Youth

This project implements the stable decision tree algorithm based on the method outline in "Improving Stability in Decision Tree Models"[^1] that presentas a unique distace metric for heuritic-based decision trees as a measure of stability. The algorithm produces a Pareto optimal set from which a single final optimal tree is selected according to an objective function targeting a unique metric to optimize (AUC, distance, combined, etc.). Our Implementation attempts to improve upon previous work[^2] in creating an effective method to identify suicide risk among youth experiencing homelessness(YEH). The dataset used in this implementation presents a unique contribution to considering social network features as well as individual factors in building risk profiles.

The distance metric implementation used in the code may be found as a reference below.[^3]


[^1]: [Improving Stability in Decision Tree Models](https://arxiv.org/abs/2305.17299)

[^2]:["Getting to the Root of the Problem: A Decision-Tree Analysis for Suicide Risk Among Young People Experiencing Homelessness"](https://doi.org/10.1086/715211)

[^3]: [Path Distance Metric Repository from Stable Decision Tree Algorithm](https://github.com/vvdigalakis/dt-distance)

## Commands to run
```bash
uv run src/StableTree/main.py --group-name FINAL_aggregate_output --option experiment --datasets data/DataSet_Combined_SI_SNI_Baseline_FE.csv data/DataSet_Combined_SI_SNI_Baseline_FE.csv data/breast_cancer.csv --labels suicidea suicattempt target

uv run src/StableTree/main.py --group-name final_aggregate_output_all_datasets --option plot --datasets data/DataSet_Combined_SI_SNI_Baseline_FE.csv data/breast_cancer.csv     
```


### Setup python & env

1. install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh 33 
brew install graphviz #graphviz binaries for pydotplus
```

2. `cd` to source directory; file(s) using UV
```
cd suicide_project
uv venv # only first time
source /bin/activate
uv sync
uv run run8.py
uv run run9.py
```


  ## Terminal Example:
==================================================
Running for dataset DataSet_Combined_SI_SNI_Baseline_FE with seed 42

==================================================
ds_nameDataSet_Combined_SI_SNI_Baseline_FE.csv

Experiment: experiment_20250501_134848_seed_42_DataSet_Combined_SI_SNI_Baseline_FE_suicidea - Seed: 42 - Dataset: DataSet_Combined_SI_SNI_Baseline_FE

Number of samples in the full dataset: 586

Number of samples in the training set: 726

Number of samples in the test set: 242

Shape of training set: (726, 56)

Shape of random split: (363, 56), (363,)

Number of trees in T0: 20

Number of trees in T: 20

Computing average tree distances |████████████████████████████████████████| 20/20 [100%] in 20.7s (0.96/s) 

Number of distances computed: 20

Average AUC score: 0.821854723038044

Number of Pareto optimal trees: 7

Frequenicies of top 2 common features: [[('trauma_sum', 70.0), ('fight', 20.0)], [('harddrug_life', 45.0), ('exchange', 15.0)], [('LEAF_NODE', 25.0), ('harddrug_life', 20.0)]]

Selected stability-accuracy trade-off final tree index: 1

Stability-accuracy tree depth: 4, nodes: 23

Selected AUC maximizing tree index: 1

AUC-maximizing tree depth: 4, nodes: 23

Selected distance minimizing tree index: 15

Distance-minimizing tree depth: 11, nodes: 79

Completed experiment: experiment_20250501_134848_seed_42_DataSet_Combined_SI_SNI_Baseline_FE_suicidea

## References:
