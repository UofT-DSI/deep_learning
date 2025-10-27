# Module 2 Summary — Neural Networks and Backpropagation

### Deep Learning — Data Sciences Institute

---

## 🧩 Overview
Module 2 introduced the inner workings of neural networks — how they are structured, trained, and optimized.  
It connected the concepts of linear models (like logistic regression) to multilayer feedforward networks,  
showing how backpropagation enables learning through gradient descent.

---

## 🧠 Key Concepts

### 1. Neural Network Basics
- A **neural network** is a differentiable function that maps inputs X to outputs Ŷ via layers of weighted linear transformations and nonlinear activations.
- Each neuron computes:
  - $z = W \cdot x + b$
  - $a = f(z)$
- A single-layer network (no hidden layers) = **Logistic Regression**.
- Adding hidden layers gives the model the capacity to represent **nonlinear** relationships.

---

### 2. Activation Functions

Activation functions introduce **nonlinearity** so that stacked layers don’t collapse into a single linear transformation.

| Function | Formula | Range | Notes |
|-----------|----------|--------|-------|
| **Sigmoid** | $\frac{1}{1 + e^{-x}}$ | $(0, 1)$ | Smooth, historically important, used for binary outputs |
| **Tanh** | $\tanh(x)$ | $(-1, 1)$ | Zero-centered version of sigmoid |
| **ReLU** | $\max(0, x)$ | $[0, ∞)$ | Modern default, avoids vanishing gradients |
| **Softmax** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0, 1)$, sums to 1 | Used for multi-class outputs |

**Sigmoid's Derivative**
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

Essential for **backpropagation**, as it determines how error signals flow backward through layers.

---

### 3. Forward Pass
Each layer performs:
$$Z^{(l)} = X^{(l-1)}W^{(l)} + b^{(l)}$$
$$A^{(l)} = f(Z^{(l)})$$

At the final layer, apply **softmax** for classification:
$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Softmax ensures outputs are valid **probabilities** — each $p_i \in [0, 1]$ and $\sum_i p_i = 1$.

---

### 4. Loss Function — Negative Log-Likelihood (Cross-Entropy)
For classification, the **Negative Log-Likelihood (NLL)** quantifies how confident the model is in the correct class:
$$\text{NLL} = -\frac{1}{N} \sum_i \log(p_{i,y_i})$$

If the model assigns low probability to the correct class, NLL is high.  
Typical ranges:
- **Good**: < 0.5  
- **Random guessing**: ~2.3 for 10 classes  
- **Poor/confused**: > 4.0

In code:
```python
eps = 1e-10
nll = -np.sum(y_onehot * np.log(y_pred + eps)) / X.shape[0]
```

---

### 5. Backpropagation and Gradients

Training minimizes loss by updating weights using **gradient descent**:
$$W \leftarrow W - \eta \frac{\partial L}{\partial W}$$

Gradients are computed using the **chain rule**:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial A^{(l)}} \cdot \frac{\partial A^{(l)}}{\partial Z^{(l)}} \cdot \frac{\partial Z^{(l)}}{\partial W^{(l)}}$$

Backprop ensures each layer learns how to adjust its weights based on its contribution to the error.

**Key takeaway:**
- **Sigmoid and dsigmoid** are used to compute how much each neuron’s output affects the total loss.
- **dsigmoid** = essential link between the forward computation and the backward gradient flow.

---

### 6. Gradient Descent and Learning Rate

Learning rate $\eta$ determines the size of each step:
- **Too high:** unstable or divergent updates  
- **Too low:** very slow convergence  

Use the rule of thumb:
```python
learning_rates = [1.0, 0.1, 0.01, 0.001]
```
and visualize **loss vs epoch** to find the stable region.

---

## 🔢 Implementation: Logistic Regression

- Logistic regression is a special case of a neural network with **no hidden layer**.
- The forward pass uses **softmax** to produce class probabilities.
- The loss uses **negative log-likelihood**.
- Gradients of loss w.r.t. weights and biases are computed for updates.

Example snippet:
```python
Z = np.dot(X, self.W) + self.b
P = softmax(Z)
loss = -np.sum(y_onehot * np.log(P + 1e-10)) / X.shape[0]
```

---

## 🧮 The Identity Matrix and One-Hot Encoding

A one-hot encoded vector is a row of the **identity matrix**.

Example for 4 categories:

$$
I_4 =
\begin{array}{cccc}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{array}
$$

Each sample’s one-hot representation corresponds to one row of this identity matrix.



**Key Insight**
Multiplying a one-hot vector by a weight matrix $W$ is the same as **selecting a row of $W$**.

$$I_K[\text{indices}] \cdot W = W[\text{indices}]$$

TensorFlow and PyTorch exploit this:  
`tf.nn.embedding_lookup(W, indices)` performs **direct row selection** instead of an expensive matrix multiply.  
No need to pre-organize categories — TensorFlow handles the identity mapping internally.

**Result:**  
Same math, but **100,000× faster** for large vocabularies.

---

## 🧠 Feedforward Multilayer Networks

Once you add a **hidden layer**, your network can model nonlinear patterns.

Example (one hidden layer):

$$Z_1 = XW_1 + b_1$$

$$A_1 = \sigma(Z_1)$$

$$Z_2 = A_1W_2 + b_2$$

$$\hat{Y} = \text{softmax}(Z_2)$$

Adding **a second hidden layer** deepens the model:

$$Z_2 = A_1W_2 + b_2 \rightarrow A_2 = \sigma(Z_2)$$

$$Z_3 = A_2W_3 + b_3 \rightarrow \hat{Y} = \text{softmax}(Z_3)$$

Backpropagation extends naturally to multiple layers.

---

### Why Use Sigmoid / dsigmoid
- Sigmoid squashes neuron outputs between 0 and 1, making them interpretable and bounded.
- dsigmoid ensures gradients can flow backward smoothly.
- However, sigmoid can cause **vanishing gradients** in deep networks — modern alternatives (ReLU, LeakyReLU) mitigate this.

---

## 🔍 Evaluating Model Predictions

To find where the model performs poorly, compute **per-sample loss**:
$$L_i = -\log(p_{i,y_i})$$

NumPy approach:
```python
probs = model.forward(X_test)
true_class_probs = probs[np.arange(len(y_test)), y_test]
sample_losses = -np.log(true_class_probs + 1e-10)
worst_indices = np.argsort(sample_losses)[::-1][:5]
```

Visualizing the “worst” samples often reveals:
- Ambiguous data
- Model overconfidence
- Systematic misclassifications

---

## ⚙️ Hyperparameter Tuning

### Parameters to experiment with
| Hyperparameter | Role |
|----------------|------|
| Learning rate ($\eta$) | Step size in gradient descent |
| Hidden layer size | Model capacity |
| Number of hidden layers | Depth and feature hierarchy |
| Epochs | Number of passes over the data |

### Experiment strategy
```python
learning_rates = [0.1, 0.01]
hidden_sizes = [32, 64]
layers = [1, 2]
```

Evaluate combinations → record train/test accuracy.

---

## 📊 Visualizing Learning Curves

Plot **loss** and **accuracy** across epochs for each configuration:
```python
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Test Loss")
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Test Accuracy")
```

- Diverging train/test accuracy → **overfitting**
- Flat low accuracy → **underfitting**
- Smooth decrease → good learning

---

## ✅ Summary Takeaways

| Concept | Key Idea |
|----------|-----------|
| **Softmax** | Converts logits to normalized probabilities |
| **Cross-Entropy / NLL** | Penalizes low confidence in true class |
| **Sigmoid + dsigmoid** | Enables nonlinear modeling and gradient flow |
| **Feedforward Network** | Layered linear + nonlinear transformations |
| **Backpropagation** | Efficient gradient computation via chain rule |
| **Identity Matrix & One-Hot** | Mathematical equivalence to embedding lookup |
| **Learning Rate** | Controls convergence stability |
| **Hidden Layers** | Increase capacity, but risk overfitting |
| **Worst Predictions** | Found via per-sample NLL |
| **TensorFlow Optimization** | Automatically replaces one-hot × W with fast lookup |

---

## 🧭 In My Own Words
> Module 2 taught me how a neural network actually “learns” — by chaining simple linear and nonlinear transformations and adjusting weights to reduce loss via backpropagation.  
> I learned that the sigmoid and softmax functions are not arbitrary — they make the math of probabilities and gradients work together smoothly.  
> My exploration of the identity matrix and one-hot encoding showed how a simple linear algebra insight translates to massive speed improvements in frameworks like TensorFlow.  
> Finally, experimenting with hyperparameters helped me see how depth, width, and learning rate interact to control both training behavior and generalization.
