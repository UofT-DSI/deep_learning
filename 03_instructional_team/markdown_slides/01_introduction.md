---
marp: true
theme: dsi_certificates_theme
paginate: true
---

# Deep Learning: Introduction to Deep Learning

```
$ echo "Data Sciences Institute"
```

---

# Warning: This lecture is more theoretical compared to the other lectures.

---

# Goal of the class

## Overview

* When and where to use DL
* "How" it works
* Frontiers of DL

## Using DL

* Implement using `Numpy`, and `Tensorflow` (`Keras`)
* Engineering knowledge for building and training DL

---
# What is Deep Learning

* Good old Neural Networks, with more layers/modules
* Non-linear, hierarchical, abstract representations of data
* Flexible models with any input/output type and size
* Differentiable Functional Programming

---
# Why Deep Learning Now?

* Better algorithms & understanding

---
# Why Deep Learning Now?

* Computing power (GPUs, TPUs, ...)

![w:700](./images/01_gpu_tpu.png)
*GPU and TPU*

---
# Why Deep Learning Now?

* Data with labels

![](./images/01_ng_data_perf.png)
*Adapted from Andrew Ng*

---
# Why Deep Learning Now?

* Open source tools and models

![w:800](./images/01_frameworks.png)

---
# DL Today: Speech-to-Text

![w:800](./images/01_speech.png)

---
# DL Today: Vision
![w:800](./images/01_vision.png)

---
# DL Today: Vision
![w:800](./images/01_vision2.png)

---
# DL Today: NLP
![w:800](./images/01_nlp.png)

---
# DL Today: NLP
![w:800](./images/01_nlp2.png)


---
# DL Today: Vision + NLP

![w:800](./images/01_nlp_vision.png)

---
# DL Today: Image translation

![w:800](./images/01_vision_translation.png)

---
# DL Today: Generative models

![w:800](./images/stackgan.jpg)

StackGAN v2 [Zhang 2017]

---
# DL Today: Generative models

Guess which one is generated?

![w:800](./images/nano_banana.png)

---
# DL Today: Generative models

![w:800](./images/claude_code.png)

---

# DL in Science: Genomics

![w:800](./images/01_deepgenomics.png)

---

# DL in Science: Genomics

![w:800](./images/protein_fold.gif)

[AlphaFold by DeepMind](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)

---

# DL in Science: Chemistry, Physics

![w:800](./images/01_deep_other.png)

---

# DL in Science: Chemistry, Physics

![](./images/Accelerating_Eulerian_Fluid_Simulation_with_Convolutional_Networks.gif)

Finite element simulator accelerated (~100 fold) by a 3D convolutional network

---
# DL for AI in games

![w:800](./images/01_games.png) 

AlphaGo/Zero: Monte Carlo Tree Search, Deep Reinforcement Learning, self-play 

---
# Outline of the class

* Backpropagation
* Computer Vision
* Recommender Systems
* Natural Language Processing
* Optimization: theory, methods and tricks
* Generative models & unsupervised learning

---

# How this course works works

* Lectures ~1 hour
* Break ~15 minutes
* Practical session ~1 hour
    * Work in breakout groups and discuss!
    * Homework: complete the lab
* Two assignments
    * One due at the end of week 1, one at the end of week 2

---

# Frameworks and Computation Graphs

---
# Libraries & Frameworks

![w:800](./images/01_frameworks.png)

This lecture is using **Keras**: high level frontend for **TensorFlow** (and MXnet, Theano, CNTK)

One lab will be dedicated to a short **Pytorch** introduction.

---
# Computation Graph

![w:700](./images/computation_graph_simple_f.png)

Neural network = parametrized, non-linear function

---
# Computation Graph

![w:700](./images/computation_graph_simple.png)

Computation graph: Directed graph of functions, depending on parameters (neuron weights)

---

# Computation Graph

![w:700](./images/computation_graph_simple_expl.png)

Combination of linear (parametrized) and non-linear functions

---

# Computation Graph

![w:700](./images/computation_graph_complicated.png)

Not only sequential application of functions

---

# Computation Graph

![](./images/computation_graph_backprop.png)
* Automatic computation of gradients: all modules are **differentiable**!
* Theano (now Aesara), **Tensorflow 1**, etc. build a static computation graph via static declarations.
* **Tensorflow 2**, **PyTorch**, **JAX**, etc. rely on dynamic differentiable modules: "define-by-run".
* Vector computation on **CPU** and accelerators (**GPU** and **TPU**).

---

# Computation Graph

![](./images/computation_graph_backprop.png)

Simple keras implementation

```py
model = Sequential()
model.add(Dense(H, input_dim=N))  # defines W0
model.add(Activation("tanh"))
model.add(Dense(K))               # defines W1
model.add(Activation("softmax"))
```

---

# Next: Lab 1!
