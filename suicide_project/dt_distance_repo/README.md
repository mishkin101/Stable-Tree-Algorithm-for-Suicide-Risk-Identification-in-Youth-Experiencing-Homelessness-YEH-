<div align="center">

# Decision Tree Distance Calculator

[![Paper](https://img.shields.io/badge/arXiv-2211.11747-red)]([https://arxiv.org/abs/2311.13695](https://arxiv.org/abs/2305.17299))
[![PyPI package](https://badge.fury.io/py/backbone-learn.svg)]([https://pypi.org/project/backbone-learn/](https://pypi.org/project/dt-distance/))

</div>

### Overview
This Python package calculates the distance between two decision trees, facilitating model comparison and analysis. 
It implements the distance metric outlined in "Improving Stability in Decision Trees Models" by Bertsimas and Digalakis Jr ([arXiv:2305.17299](https://arxiv.org/abs/2305.17299)https://arxiv.org/abs/2305.17299).

## Getting Started

### Installation

Install dt-distance using pip:
```python
pip install dt-distance
```

### Usage
```
from dt_distance.distance_calculator import DistanceCalculator 
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

# Train trees
tree1 = DecisionTreeClassifier(max_depth=1, random_state=0)
tree2 = DecisionTreeClassifier(max_depth=2, random_state=1)
tree1.fit(X, y)
tree2.fit(X, y)

# Initialize the DistanceCalculator with the two trees and the dataset
distance_calculator = DistanceCalculator(tree1, tree2, X=X, y=y)

# Compute the distance
distance_calculator.compute_tree_distance()

print(f"The distance between the two decision trees is: {distance_calculator.distance}")

print(f"The optimal path matching is: {distance_calculator.matching}")
```

### Main Parameters
- tree1, tree2: Two sklearn DecisionTreeClassifier objects.
- problem_params (optional): Specifies feature info. Defaults to inferred from trees if not provided.
- X, y (optional): Data matrices to infer feature info - if problem_params is not used.
- max_depth (optional): For normalization, defaults to the deeper of the two trees.


## Citation
If using this tool in your research, please cite the associated paper:
```
@misc{bertsimas2023improving,
      title={Improving Stability in Decision Tree Models}, 
      author={Dimitris Bertsimas and Vassilis Digalakis Jr au2},
      year={2023},
      eprint={2305.17299},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
