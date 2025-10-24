# Module 3: Recommender Systems & Embeddings — Deep Learning Notes (Editable Canvas)

---

## Overview
Module 3 introduces **embeddings** and their role in **recommender systems**, alongside **dropout regularization** to mitigate overfitting. It explores how neural networks process categorical inputs by embedding them in continuous vector spaces.

> 📝 **Your Notes:**
> 
> 
---

## 1. Embeddings

### From Real to Symbolic Inputs
- Real-valued inputs are straightforward for models.
- Symbolic data requires transformation into numeric form (e.g., words, user IDs, product categories).

> 📝 **Your Notes:**
> 

### One-Hot Representation
- Large sparse vectors where each dimension represents one symbol.
- Equal distances between symbols — no semantic information.

> 📝 **Your Notes:**
> 

### Embeddings
- Dense, continuous vectors learned via backpropagation.
- Captures **semantic relationships** (e.g., similarity, context).
- Embedding matrix shape: |V| × d.

> 📝 **Your Notes:**
> 

> **Quote:** *Yann LeCun:* “Computing symbolic operations in algebraic space makes it possible to optimize via gradient descent.”

### Distance and Similarity Metrics
- **Euclidean Distance:** measures absolute distance but depends on scale.
- **Cosine Similarity:** focuses on direction, invariant to vector magnitude.

> 📝 **Your Notes:**
> 

### Visualization
- **PCA:** linear dimensionality reduction.
- **t-SNE:** non-linear method preserving local structure (typical perplexity: 20–30).

> 📝 **Your Notes:**
> 

---

## 2. Dropout Regularization

### Overfitting
- When a model performs well on training but poorly on unseen data.
- Detected when training loss decreases while validation loss increases.

> 📝 **Your Notes:**
> 

### Regularization Methods
- Limit network width/depth.
- Add **L2 penalty** on weights.
- Use **Dropout** to reduce dependency on specific neurons.

### How Dropout Works
- Randomly sets activations to zero with probability *p* during training.
- Promotes redundancy and robustness.
- Acts as noise injection and implicit ensemble learning.

**Example (Keras):**
```python
model = Sequential([
    Dense(hidden_size, input_shape=(input_dim,), activation='relu'),
    Dropout(0.5),
    Dense(hidden_size, activation='relu'),
    Dropout(0.5),
    Dense(output_size, activation='softmax')
])
```

> 📝 **Your Notes:**
> 

---

## 3. Recommender Systems

### Applications
- Streaming platforms: Netflix, Spotify, YouTube.
- E-commerce: Amazon, Etsy.
- Social media: feed ranking, ad targeting.

> 📝 **Your Notes:**
> Value placed in commerce on these systems working well

### Core Approaches
1. **Content-Based Filtering:** recommends based on item attributes.
2. **Collaborative Filtering (CF):** recommends based on user-item interactions. Measured over time rather than explicitly declared for data points
3. **Hybrid Models:** blend both to handle cold-start and sparse data.

> 📝 **Your Notes:**
> 

### Feedback Types
- **Explicit:** user-provided ratings (stars, likes). Metrics: RMSE, MAE.
- **Implicit:** inferred preferences (clicks, watch time). Metrics: ROC-AUC, Precision@K, NDCG.

> 📝 **Your Notes:**
> 

---

## 4. Ethical Considerations

### Bias and Fairness
- Algorithms can amplify biases and polarization.
- Examples: gender bias in job ads, filter bubbles in recommendations.

> 📝 **Your Notes:**
> 

### Mitigation Strategies
- Avoid sensitive inputs (e.g., gender, name).
- Use fairness tools: **SHAP**, **Integrated Gradients**, **Fairlearn**.
- Ensure transparency and explainability.

> 📝 **Your Notes:**
> 

**References:**
- Edwards & Storkey (2016), *Censoring Representations with an Adversary*
- [Data Transparency Lab](http://www.datatransparencylab.org/)
- [TransAlgo Initiative (France)](https://transalgo.irisa.fr)

---

## 5. Lab 3 Summary — Embeddings and Recommendations in Practice

### Goal
Implement and visualize user–item embeddings using TensorFlow/Keras.

### Steps
1. **Data Preparation** – Encode user and item IDs.
2. **Model Architecture** – Build dual embedding layers and combine them via a dot product.
   ```python
   user_input = Input(shape=(1,))
   item_input = Input(shape=(1,))
   user_vec = Embedding(num_users, emb_dim)(user_input)
   item_vec = Embedding(num_items, emb_dim)(item_input)
   dot = Dot(axes=2)([user_vec, item_vec])
   output = Dense(1, activation='sigmoid')(Flatten()(dot))
   model = Model([user_input, item_input], output)
   ```
3. **Training** – Use binary cross-entropy; test different dropout rates.
4. **Visualization** – Project embeddings via t-SNE/PCA to inspect structure.
5. **Evaluation** – Metrics: Precision@K, Recall@K, NDCG.

> 📝 **Your Notes:**
> 

### Learning Outcomes
- Understand latent representations of users/items.
- Build a deep collaborative filtering model.
- Visualize and interpret learned embeddings.

> 📝 **Your Notes:**
> 

---

## Key Takeaways
- **Embeddings:** transform symbols into vector spaces.
- **Dropout:** regularizes models by introducing noise.
- **Recommender Systems:** leverage embeddings to model user–item relationships.
- **Ethics:** fairness and transparency must guide deployment.

> 📝 **Your Notes:**
> 

---

**Next:** Proceed to *Lab 3 – Implementing embeddings and recommender system components.*

