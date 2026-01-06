# Mini Neural Network with Python and uv

This project demonstrates how to build a simple neural network from scratch using **NumPy** and manage the project with **uv**.

## Project Structure

- `neural_network.py`: Contains the `MiniNeuralNetwork` class with forward and backward propagation.
- `test_nn.py`: A script that generates a sample dataset (points in a circle) and trains the network.
- `pyproject.toml`: Project configuration managed by `uv`.

## How to Run

Ensure you have `uv` installed. Then run:

```bash
uv venv && source .venv/bin/activate
uv sync
uv run test_nn.py
```

## Step-by-Step Explanation

### 1. Initialization (`__init__`)

We initialize weights with small random values. If weights are all zero, the neurons will learn the same features (symmetry problem).

- **W1**: Weights between input and hidden layer.
- **W2**: Weights between hidden and output layer.

### 2. Forward Propagation (`forward`)

The network calculates the output by passing inputs through the layers:

1. **Linear Transformation**: $Z = X \cdot W + b$
2. **Activation**: $A = \sigma(Z)$ (Sigmoid function)
   The Sigmoid function squashes values between 0 and 1, which is useful for probability-like outputs.

### 3. Backward Propagation (`backward`)

This is where the "learning" happens using **Gradient Descent**:

1. **Calculate Error**: The difference between target and prediction.
2. **Chain Rule**: We calculate how much each weight contributed to the error.
3. **Update Weights**: We subtract a portion of the gradient (scaled by `learning_rate`) from the weights to reduce the error in the next pass.

### 4. Training (`train`)

We repeat the forward and backward passes for many iterations (epochs) until the loss (error) is minimized.

## Sample Dataset

The `test_nn.py` script creates a dataset of 200 points. Points inside a circle of radius 0.7 are labeled `1`, and points outside are labeled `0`. This is a non-linear classification problem that a simple linear model cannot solve, but our mini neural network can!
