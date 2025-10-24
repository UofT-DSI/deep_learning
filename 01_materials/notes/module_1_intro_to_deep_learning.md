# 🧠 Module 1: Introduction to Deep Learning

## Overview
**Goals of the class:**
- Understand **when and where** to use Deep Learning  
- Learn **how** it works — from theory to implementation  
- Explore the **frontiers** of DL research  

**Tools:** NumPy, TensorFlow, Keras  

---

## What Is Deep Learning?
- Deep Learning = Neural Networks with **many layers**  
- Builds **hierarchical**, **nonlinear**, **abstract** data representations  
- Works with any input/output type or size  
- Implemented as **differentiable functions** (supports gradient descent optimization)

---

## Why Deep Learning Now?
1. **Algorithms & Understanding** – improved optimization, activations, architectures  
2. **Compute Power** – GPUs, TPUs  
3. **Big Labeled Datasets** – ImageNet, text corpora, etc.  
4. **Open Source Ecosystem** – PyTorch, TensorFlow, JAX, pretrained models  

---

## Applications
- **Speech**: speech-to-text (e.g., DeepSpeech)  
- **Vision**: object recognition, image translation  
- **NLP**: translation, chatbots, summarization  
- **Generative Models**: GANs, diffusion models, audio synthesis (WaveNet, Tacotron 2)  
- **Science**: AlphaFold (protein folding), physics simulators  

---

## Deep Learning in Practice
**Frameworks & Computation Graphs**
- A neural network = a *parameterized nonlinear function*  
- Composed of **linear + nonlinear** functions  
- Modern frameworks (TF2, PyTorch, JAX) use *define-by-run* dynamic computation graphs  
- Example:
  ```python
  model = Sequential()
  model.add(Dense(H, input_dim=N))
  model.add(Activation("tanh"))
  model.add(Dense(K))
  model.add(Activation("softmax"))
  ```

---

## Outline of Course Modules
1. Backpropagation  
2. Computer Vision  
3. Recommender Systems  
4. Natural Language Processing  
5. Optimization Theory  
6. Generative & Unsupervised Learning  

---

## Note Area
Use this section to jot down questions or insights as you study.

**Notes:**
- 📝 *DL = Differentiable Programming: each layer defines a function and gradients flow through them.*  
- 🧩 *Think in terms of computation graphs rather than “layers” only.*

**Questions:**
- What are the practical limits of using DL for small datasets?  
- How does computation graph design influence model performance and debugging?  

