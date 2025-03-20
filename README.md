## suicide_project

## useful links

### Trees

+ https://youtu.be/_L39rN6gz7Y?si=9QgZsz_NEvCzbNvK
+ https://koalaverse.github.io/machine-learning-in-R/decision-trees.html 
+ https://stat2labs.sites.grinnell.edu/Handouts/rtutorials/ClassificationTrees.html
+ https://arxiv.org/pdf/1706.09773

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

## R stuff

[ ] DataExplorer package to summarize the dataset (https://boxuancui.github.io/DataExplorer/)