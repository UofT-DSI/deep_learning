# Module 2: Neural Networks and Backpropagation
**Deep Learning — Data Sciences Institute**

---

## Overview
- How neural networks are structured and trained  
- The role of backpropagation in adjusting weights  
- How loss, gradients, and optimization interact  
- Key practical tricks: initialization, normalization, optimizers

---

## Key Concepts

### Neural Network Basics
- A neural network parametrizes a **conditional distribution** \( P(Y|X) \)
- **Neuron:** linear transformation + non-linear activation  
  \( z = W \cdot x + b \)  
  \( f(x) = \sigma(z) \)
- **Layers:** multiple neurons connected in matrix form
- **MLP (Multi-layer Perceptron):** feedforward network with one or more hidden layers

> **Note:** Logistic regression = MLP without hidden layer

---

### Activation Functions
Common examples:
- **ReLU:** \( \text{ReLU}(x) = \max(0, x) \)
- **Tanh:** \( \tanh(x) \)
- **Sigmoid:** \( \frac{1}{1 + e^{-x}} \)
- **Softmax:** outputs class probabilities that sum to 1

**Question:** When would ReLU outperform Tanh or Sigmoid?

---

### Loss Functions
- **Cross-Entropy Loss** for classification  
  \( L = - \sum y_i \log(\hat{y}_i) \)
- **Mean Squared Error (MSE)** for regression
- **Regularization:** L2 penalty (weight decay)  
  \( \lambda \|W\|^2 \)

> **Note:** Regularization = Maximum A Posteriori (MAP) estimate

---

### Gradient Descent
Minimize the loss by updating parameters:
\[
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
\]

Variants:
- **Batch Gradient Descent** – full dataset
- **Stochastic Gradient Descent (SGD)** – single sample
- **Mini-batch SGD** – subset of samples

> **Tip:** Learning rate \( \eta \) controls convergence speed and stability.

---

### Backpropagation
- Uses the **chain rule** to compute gradients layer by layer
- Key quantities:
  - \( \nabla_{W^o} L = (\hat{y} - y) \cdot h(x)^T \)
  - \( \nabla_{b^o} L = \hat{y} - y \)
  - \( \nabla_{W^h} L = \text{propagate backward via chain rule} \)
- Backprop through the network computes gradients for all weights efficiently

**Question:** Why does random initialization matter for gradient flow?

---

### Initialization and Normalization
- Input features should be **normalized**
- Weight initialization should avoid symmetry:
  - \( W \sim \mathcal{N}(0, 0.01) \)
  - **Xavier** or **He** initialization improves gradient stability
- Biases often initialized to zero

> **Observation:** Poor initialization = vanishing/exploding gradients

---

### Optimization Tricks
- **Learning rate scheduling:** reduce when validation loss plateaus  
  `ReduceLROnPlateau` in Keras
- **Momentum:** accumulate previous gradients  
  \( v_t = \gamma v_{t-1} + \eta \nabla_\theta L \)
- **Adam optimizer:** adaptive learning rates per parameter
  - Good default: `lr=3e-4`
  - Combines momentum and adaptive updates

---

### Practical Considerations
- Normalize inputs
- Use small learning rates initially
- Monitor training and validation loss
- Experiment with optimizers: SGD, Adam, RMSProp
- Clip gradients to prevent explosion

---

## Takeaways
- Backpropagation = efficient application of chain rule
- Initialization and normalization crucial for learning
- Momentum and Adam accelerate convergence
- Overfitting requires regularization or dropout (next module!)

---

## Notes & Questions
🟢 *Your notes here...*  
💡 *Question:* What conditions make Adam outperform SGD?  
🧩 *Experiment idea:* Implement an MLP with ReLU and compare gradient flow to Tanh.

