# Neural Network Forward & Backward Propagation

A NumPy-based implementation of a small neural network to demonstrate **forward propagation, loss calculation, backward propagation, and gradient descent** from scratch.

The notebook focuses on understanding the mathematical flow of neural network training without relying on TensorFlow or PyTorch for the forward and backward passes.

## Network Architecture

The network consists of:

```text
Input Layer
    ↓
Hidden Layer 1
    ↓
Hidden Layer 2
    ↓
Output Layer
```

The input contains two values:

```text
X = [1.0, -2.0]
```

The network uses:

- ReLU activation in the hidden layers
- Linear activation at the output
- MSE-based loss
- Learning rate of `0.01`
- Bias value initialized to `1`

## Forward Propagation

The notebook calculates:

1. Hidden Layer 1 weighted sum
2. ReLU activation
3. Hidden Layer 2 weighted sum
4. ReLU activation
5. Output layer prediction
6. Loss

The forward pass demonstrates how the input moves through the network to produce a prediction.

## Backward Propagation

The backward pass calculates gradients for:

- Output layer
- Hidden Layer 2
- Hidden Layer 1

The implementation uses the derivative of ReLU to propagate the error backward through the network.

## Weight Update

After calculating the gradients, the weights are updated using gradient descent:

```text
new_weight = old_weight - learning_rate × gradient
```

The notebook performs one weight-update step using a learning rate of `0.01`.
