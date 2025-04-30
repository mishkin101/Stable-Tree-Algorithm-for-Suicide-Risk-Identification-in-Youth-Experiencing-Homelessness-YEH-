## suicide_project

potential fixes for quarto rendering issues:
```bash
uv venv
source .venv/bin/activate
uv sync
uv run src/StableTree/main.py --seeds 23 28 --group-name seed_checking
```

- trimmed tree in logs/experiment_name/
- replication of anthony's plot style (feature importance plot, dt plot & metrics.txt) in outputs/
- call using `uv run src/StableTree/main.py`

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

[ ] Install R `brew install r`
[ ] Quarto (https://quarto.org/docs/get-started/)
[ ] VS Code extensions
    [ ] Quarto
    [ ] R
[ ] run `main.qmd`

## notes

- is this the best nethod in lit.?
- robust to distribution shifts? numerical diff in distribution
- https://arxiv.org/abs/2310.17772
- https://optimization-online.org/wp-content/uploads/2023/10/RobustTrees_arXiv-2.pdf

