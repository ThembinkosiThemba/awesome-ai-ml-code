# Project 05: Decision Tree from Scratch

This project implements a **Decision Tree Classifier** from scratch using Python and NumPy. It is a fundamental supervised learning algorithm used for classification tasks.

## How it Works

A Decision Tree works by recursively splitting the data into subsets based on the feature that provides the most "information" at each step. The goal is to create branches that lead to pure leaf nodes (nodes where all samples belong to the same class).

### 1. Entropy

Entropy is a measure of impurity or randomness in a dataset. For a binary classification task, it is calculated as:
$$H(y) = - \sum p(y) \log(p(y))$$
Where $p(y)$ is the probability of a class appearing in the dataset. A dataset with only one class has an entropy of 0, while a perfectly balanced dataset has an entropy of 1.

### 2. Information Gain

Information Gain is the reduction in entropy achieved by splitting the data on a specific feature. We calculate the entropy of the parent node and subtract the weighted average entropy of the children nodes:
$$IG = \text{Entropy(parent)} - [\text{Weighted Avg} \times \text{Entropy(children)}]$$
The algorithm chooses the feature and threshold that maximize this gain.

### 3. Recursive Splitting

The tree grows by repeating this process for each child node until a stopping criterion is met, such as:

- Reaching the maximum depth.
- Having fewer than the minimum required samples to split.
- The node becoming perfectly pure (all samples are the same class).

## Project Structure

| File               | Description                                                                                                                    |
| :----------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| `decision_tree.py` | Contains the `Node` and `DecisionTree` classes. Implements entropy calculation, information gain, and recursive tree building. |
| `main.py`          | Generates a synthetic dataset (Age vs. Income for product purchase) and trains/tests the model.                                |
| `pyproject.toml`   | Project configuration managed by `uv`.                                                                                         |

## How to Run

Ensure you have `uv` installed, then run:

```bash
uv run main.py
```

## Sample Dataset

The `main.py` script generates a dataset of 100 people with two features: **Age** and **Income**. The target is whether they "Buy a Product". The logic is that older people with higher incomes are more likely to buy, but we've added some noise to make it a realistic challenge for the tree.
