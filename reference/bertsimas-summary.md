# stability - bertsimas

## 1. Overview of the Proposed Method

1. **Train a “first round” of decision trees** on an initial dataset $X_{0}$.  
2. **Train a “second round” of decision trees** on an augmented dataset $(X_{0}, X_{1}) = X$.  
3. For each tree in the second round, **compute its average distance** to all trees in the first round to measure its stability.  
4. **Compute predictive metrics** (e.g., accuracy or AUC on a holdout set).  
5. **Identify the Pareto frontier** that highlights the trade-off between stability and accuracy.  
6. **Choose a single tree** from the Pareto frontier based on user preferences (e.g., weighting stability vs. accuracy).  

The next sections provide more details on each step, including the mathematical definitions, data processing details, and considerations for real-world applications—particularly in healthcare.

---
## 2. Notation and Data Setup

- **Data notation**:  
  - $X \in \mathbb{R}^{N \times P}$ is your full feature matrix (with $N$ samples, $P$ features).  
  - $y \in [K]^{N}$ is the vector of class labels (binary or multi-class).  
  - Split this into training/testing sets (e.g., 67%/33%) and (optionally) further into $(X_{0}, X_{1})$ if you want to mimic a “data-comes-in-stages” scenario.  

- **Feature types**:  
  - $\mathcal{N} \subseteq [P]$ = set of numerical features, each with lower/upper bounds $\ell_j, u_j$.  
  - $\mathcal{C} = [P] \setminus \mathcal{N}$ = set of categorical features, each with $c_j$ categories.  

- **Algorithmic approach**:  
  - You may use any tree learning procedure (e.g., CART, optimal MIO-based trees, etc.).  
  - You will generate a *collection of candidate trees* for each dataset split, typically by varying hyperparameters (depth, minimum samples per leaf, etc.) and/or by bootstrapping.  

---

## 3. Representing a Decision Tree by Its Paths

A decision tree of depth $D$ partitions the feature space into up to $2^{D}$ “leaf” regions. Each leaf is associated with:

- Upper and lower bounds for numerical features.  
- Subsets of categories for categorical features.  
- A predicted class label $k \in [K]$.  

Each leaf can be viewed as a *path* from the root node down to a leaf node. Formally, define a path $p$ by:

1. **Bounds on numerical features**:

$$
(u_{j}^{p}, \; \ell_{j}^{p}) \quad \text{for each numerical feature } j\in \mathcal{N},
$$

where $\ell_{j}^{p}, u_{j}^{p} \in [\ell_j, u_j]$.

2. **Category subsets for categorical features**:

$$
\mathbf{c}_{j}^{p} \in \{0,1\}^{c_j} \quad \text{for each categorical feature } j\in \mathcal{C},
$$

indicating which categories are kept.

3. **A class label** $k^{p}$.

Hence, each path $p$ can be represented as

$$
p = \bigl(u^{p}, \;\ell^{p}, \;C^{p}, \;k^{p}\bigr),
$$

where $C^{p}$ stacks all the $\mathbf{c}_{j}^{p}$ vectors (one row per categorical feature).

A *decision tree* $\mathcal{T}$ is a collection of paths $\mathcal{P}(\mathcal{T}) = \{p_{1},p_{2},\dots,p_{T}\}$.

---

## 4. Distance Metric for Decision Trees

### 4.1 Distance Between Two Paths

The distance $d(p, q)$ between paths $p$ and $q$ has two parts:
1. **Overlap in feature ranges** (for numerical and categorical splits).  
2. **Discrepancy in assigned class labels**.

Concretely:

$$
d(p,q) \;=\; 
\sum_{j\in \mathcal{N}} \frac{\bigl|u_{j}^{p} \;-\; u_{j}^{q}\bigr| + \bigl|\ell_{j}^{p}\;-\;\ell_{j}^{q}\bigr|}{\,2\bigl(u_{j}-\ell_{j}\bigr)} 
\;+\;\sum_{j\in \mathcal{C}} \frac{\bigl\|\mathbf{c}_{j}^{p} - \mathbf{c}_{j}^{q}\bigr\|_{1}}{c_{j}}
\;+\;\lambda\;\mathbf{1}_{\bigl(k^{p} \neq k^{q}\bigr)},
$$

where

- $\|\mathbf{c}_{j}^{p} - \mathbf{c}_{j}^{q}\|_{1}$ is the entrywise $\ell_{1}$-norm (i.e., number of categorical indicators that differ).  
- $\mathbf{1}_{(k^{p} \neq k^{q})}$ is 1 if the two paths assign different class labels, else 0.  
- $\lambda$ is a user-defined “importance weight” that balances how much label mismatch should contribute to the path distance vs. overlap in features.

> **Practical tip**: Setting $\lambda = 2D$ often ensures that having different class labels in leaves is weighed about as much as having the entire path differ in up to $D$ splits.

### 4.2 Path Weight

A path that is not matched in one tree but “has no counterpart” in the other is given a “weight,” which measures how “large” or “important” that path’s region is in the feature space. Formally:

$$
w(p) \;=\; \sum_{j \in \mathcal{N}} \frac{(u_{j}^{p} \;-\;\ell_{j}^{p})}{(u_j - \ell_j)} \,\mathbf{1}_{(u_{j}^{p} \neq u_j \,\text{or}\,\ell_{j}^{p}\neq \ell_j)}
\;+\; \sum_{j \in \mathcal{C}} \frac{\|\mathbf{c}_{j}^{p}\|_{1}}{c_j}\,\mathbf{1}_{(\mathbf{c}_{j}^{p} \neq \text{“all categories”})}.
$$

This ensures that unmatched paths that carve out large or complex regions of the feature space incur a bigger penalty in the overall tree distance.

### 4.3 Distance Between Two Trees

Given two trees $\mathcal{T}_{1}, \mathcal{T}_{2}$ with path sets $\mathcal{P}(\mathcal{T}_{1}), \mathcal{P}(\mathcal{T}_{2})$, define $T_{1} = |\mathcal{P}(\mathcal{T}_{1})|$ and $T_{2} = |\mathcal{P}(\mathcal{T}_{2})|$. Assume $T_{1} \geq T_{2}$ w.l.o.g.

We introduce binary decision variables:

$$
x_{p,q} \;=\; \mathbf{1}\bigl(\text{path } p \text{ in } \mathcal{T}_{1}\text{ is matched with path }q\text{ in }\mathcal{T}_{2}\bigr),
\quad
x_{p} \;=\; \mathbf{1}\bigl(\text{path } p \text{ remains unmatched}\bigr).
$$

Then compute

$$
d\bigl(\mathcal{T}_{1}, \mathcal{T}_{2}\bigr) 
\;=\;\min_{\{x\}}\;
\sum_{p\in\mathcal{P}(\mathcal{T}_{1})}\sum_{q\in\mathcal{P}(\mathcal{T}_{2})} d(p,q)\, x_{p,q}
\;+\;\sum_{p\in\mathcal{P}(\mathcal{T}_{1})} w(p)\, x_{p}
$$

subject to

$$
\sum_{q\in\mathcal{P}(\mathcal{T}_{2})} x_{p,q} + x_{p} \;=\;1 \quad \forall\,p\in\mathcal{P}(\mathcal{T}_{1}),
$$

$$
\sum_{p\in\mathcal{P}(\mathcal{T}_{1})} x_{p,q} \;=\;1 \quad \forall\,q\in\mathcal{P}(\mathcal{T}_{2}),
$$

$$
x_{p,q},\,x_{p} \;\;\in\;\{0,1\}.
$$

Because this is a bipartite matching with optional “skips,” you can solve it efficiently (the linear relaxation has integral solutions). In Python, you could use a standard linear (or integer) optimization package (e.g., Gurobi, `PuLP`, or `ortools`) to solve this assignment problem.

---

## 5. Training Stable Decision Trees

### 5.1 Generating Tree Collections

1. **First collection** ($\mathcal{T}_{0}$):  
   - Train on $X_{0}$ only.  
   - For each combination of hyperparameters (e.g., `max_depth ∈ {3,5,7}`, `min_samples_leaf ∈ {3,5,10}`, …) do multiple bootstrap runs and store the resulting trees.  
   - Let $\mathcal{T}_{0} = \{\mathcal{T}_{1}^{0}, \ldots, \mathcal{T}_{B}^{0}\}$.  

2. **Second collection** ($\mathcal{T}$):  
   - Train on the *full* dataset $X = (X_{0}, X_{1})$.  
   - Use the *same set of hyperparameter combinations* and bootstrap sampling if desired.  
   - Let $\mathcal{T} = \{\mathcal{T}_{1}, \ldots, \mathcal{T}_{B}\}$.  

> **Practical note**: Usually $B$ will be in the low hundreds or possibly up to 1000, given you are enumerating over hyperparameters × bootstrap seeds.

### 5.2 Computing Stability of Each Tree

For each $\mathcal{T}_{b} \in \mathcal{T}$, define

$$
d_{b} \;=\; \frac{1}{|\mathcal{T}_{0}|} \sum_{\beta=1}^{|\mathcal{T}_{0}|} d\bigl(\mathcal{T}_{\beta}^{0},\;\mathcal{T}_{b}\bigr).
$$

- This is the *average structural distance* from $\mathcal{T}_{b}$ to *all* trees in the first collection.  
- Smaller $d_{b}$ means higher similarity to the trees you trained initially on $X_{0}$.

### 5.3 Measuring Predictive Power

- You can measure *out-of-sample performance* for each tree $\mathcal{T}_{b} \in \mathcal{T}$ in multiple ways:  
  - **Accuracy** (binary classification).  
  - **AUC** (Area Under the ROC Curve).  
  - **Log-loss**, etc.  

Let $\alpha_{b}$ be the predictive metric for $\mathcal{T}_{b}$.

### 5.4 Finding the Pareto Frontier

To explore the *trade-off between stability* $(d_{b})$ and *accuracy* $(\alpha_{b})$, do:

1. **Plot** $(d_{b}, \alpha_{b})$ for each $\mathcal{T}_{b}\in \mathcal{T}$.  
2. **Identify Pareto points**: a tree $\mathcal{T}_{b}$ is on the Pareto frontier if there is *no other* tree $\mathcal{T}_{b'}$ that *dominates* it in *both* stability and accuracy. Formally, $\mathcal{T}_{b}$ is Pareto-optimal if no $\mathcal{T}_{b'}$ satisfies

$$
d_{b'} \le d_{b} \quad\text{and}\quad \alpha_{b'} \ge \alpha_{b},
$$

with at least one strict inequality.

In practice:

- Collect the set $\mathcal{T}^{*}\subset \mathcal{T}$ of all *Pareto-optimal* trees.  
- The user can **choose** among those (e.g., purely by best accuracy, purely by best stability, or a linear combination such as $\max \{\alpha_{b} - \gamma\, d_{b}\}$ for some $\gamma>0$).  

> **Result**: You obtain a set of candidate decision trees that cleanly map out the stability–accuracy curve.

---

## 7. Additional Considerations

1. **Scaling the Distance**: You can optionally report the distance as a *percentage* of the maximum possible distance between two trees of depth $D$. The paper states a rough upper bound $ d(\mathcal{T}_1,\mathcal{T}_2) \le 2^{D}(2D + \lambda)$. Thus, you can divide your computed $d(\mathcal{T}_1,\mathcal{T}_2)$ by $2^{D}(2D + \lambda)$ to get a normalized number in $[0,1]$ or $[0,100\%]$.

2. **Multiple Data Splits**:  
   - For more robust estimates, the authors use multiple random splits (e.g., 10 replications) and report averages + standard deviations of stability and performance.

3. **Interpretability Measures**:  
   - You can examine each Pareto tree’s number of leaves, maximum depth, or average path length to see how *interpretability* interacts with stability.

4. **Feature Importance**:  
   - If desired, measure stability in feature usage by examining how often each feature is used in the first few splits, or computing the Gini-based importances for each tree.

5. **Tree Extraction**:  
   - For *optimal decision trees* from an MIO solver, you will directly have path constraints in the solver. You can still parse them post hoc in a similar manner.

6. **Complexity**:  
   - The assignment-based distance is solved in polynomial time in the number of leaves. In the worst case (maximum depth $D$ with all splits used) each tree can have up to $2^{D}$ leaves, so the bipartite matching uses an $O(2^{D} \times 2^{D})$ cost matrix. For moderate $D$ (up to 10–12 in practice), this is usually feasible.

---

## 8. Summary of Key Equations and Algorithms

1. **Path Distance**:

$$
d(p, q) 
  = \sum_{j \in \mathcal{N}} \frac{|u_j^p - u_j^q| + |\ell_j^p - \ell_j^q|}{\,2(u_j - \ell_j)} 
    + \sum_{j \in \mathcal{C}} \frac{\|\mathbf{c}_j^p - \mathbf{c}_j^q\|_1}{c_j}
    + \lambda \,\mathbf{1}_{(k^p \neq k^q)}.
$$

2. **Path Weight**:

$$
w(p) 
  = \sum_{j \in \mathcal{N}} \frac{(u_j^p - \ell_j^p)}{(u_j - \ell_j)}\,\mathbf{1}_{(\text{split on } j)}
    + \sum_{j \in \mathcal{C}} \frac{\|\mathbf{c}_j^p\|_1}{c_j}\,\mathbf{1}_{(\text{split on } j)}.
$$

3. **Distance Between Two Trees**:

$$
d(\mathcal{T}_1,\mathcal{T}_2)
  = \min_{\{x_{p,q}, x_p\}}
    \sum_{p \in \mathcal{P}(\mathcal{T}_1)}
    \sum_{q \in \mathcal{P}(\mathcal{T}_2)} d(p,q)\, x_{p,q}
    + \sum_{p \in \mathcal{P}(\mathcal{T}_1)} w(p)\, x_p
$$

subject to the bipartite matching constraints.

4. **Average Distance to a Tree Collection**:

$$
d_{b}
  = \frac{1}{\,|\mathcal{T}_0|\,}\sum_{\beta=1}^{|\mathcal{T}_0|}
    d\bigl(\mathcal{T}_\beta^{0},\;\mathcal{T}_b\bigr).
$$

5. **Pareto Frontier**:  
   - The set of all $\mathcal{T}_b$ for which no $\mathcal{T}_{b'}$ has $d_{b'} \le d_b$ *and* $\alpha_{b'} \ge \alpha_b$ with at least one strict inequality.

---
