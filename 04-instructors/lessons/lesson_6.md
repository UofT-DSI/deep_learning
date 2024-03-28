---
marp: true
theme: dsi-certificates-theme
_class: invert
paginate: true
---

# Deep Learning
## Natural Language Processing with Deep Learning

Adapted from material by Charles Ollion & Olivier Grisel

---

## Natural Language Processing

.center[
<img src="./images/nlp.png" style="width: 550px;" />
]

---

## Natural Language Processing

- Sentence/Document level Classification (topic, sentiment)
- Topic modeling (LDA, ...)
- Translation
- Chatbots / dialogue systems / assistants (Alexa, ...)
- Summarization

---

Useful open source projects

.center[
<img src="./images/logolibs.png" style="width: 500px;" />
]

---

# Outline

* Classification and word representation
* Word2Vec
* Language Modelling
* Recurrent neural networks

---

# Word Representation and Word2Vec

---

# Word representation

* Words are indexed and represented as 1-hot vectors
* Large Vocabulary of possible words $|V|$
* Use of **Embeddings** as inputs in all Deep NLP tasks
* Word embeddings usually have dimensions 50, 100, 200, 300

---

# Supervised Text Classification

.center[
<img src="./images/fasttext.svg" style="width: 500px;" />
]

.footnote.small[
Joulin, Armand, et al. "Bag of tricks for efficient text classification." FAIR 2016
]

Question: shape of embeddings if hidden size is H

---

* $\mathbf{E}$ embedding (linear projection) .right.red[`|V| x H`]
* Embeddings are averaged .right[hidden activation size: .red[`H`]]
* Dense output connection $\mathbf{W}, \mathbf{b}$ .right.red[`H x K`]
* Softmax and **cross-entropy** loss

---

# Supervised Text Classification

.center[
<img src="./images/fasttext.svg" style="width: 500px;" />
]

.footnote.small[
Joulin, Armand, et al. "Bag of tricks for efficient text classification." FAIR 2016
]

- Very efficient (**speed** and **accuracy**) on large datasets
- State-of-the-art (or close to) on several classification, when adding **bigrams/trigrams**
- Little gains from depth

---

# Transfer Learning for Text

Similar to image: can we have word representations that are generic
enough to **transfer** from one task to another?

---

**Unsupervised / self-supervised** learning of word representations

---

**Unlabelled** text data is almost infinite:
  - Wikipedia dumps
  - Project Gutenberg
  - Social Networks
  - Common Crawl

---
# Word Vectors

.center[
<img src="./images/tsne_words.png" style="width: 630px;" />
]

.footnote.small[
excerpt from work by J. Turian on a model trained by R. Collobert et al. 2008
]

Question: what distance to use in such a space

---

# Word2Vec

.center[
<img src="./images/most_sim.png" style="width: 500px;" />
]

.footnote.small[
Colobert et al. 2011, Mikolov, et al. 2013
]

---

### Compositionality

.center[
<img src="./images/sum_wv.png" style="width: 700px;" />
]

---

# Word Analogies

.center[
<img src="./images/capitals.png" style="width: 450px;" />
]

.footnote.small[
Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." NIPS 2013
]

---

- Linear relations in Word2Vec embeddings
- Many come from text structure (e.g. Wikipedia)

---
# Self-supervised training

Distributional Hypothesis (Harris, 1954):
*“words are characterised by the company that they keep”*

Main idea: learning word embeddings by **predicting word contexts**

.footnote.small[
Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." NIPS 2013
]

---

Given a word e.g. “carrot” and any other word $w \in V$ predict
probability $P(w|\text{carrot})$ that $w$ occurs in the context of
“carrot”.

---

- **Unsupervised / self-supervised**: no need for class labels.
- (Self-)supervision comes from **context**.
- Requires a lot of text data to cover rare words correctly.

How to train fastText like model on this?

---

# Word2Vec: CBoW

CBoW: representing the context as **Continuous Bag-of-Words**

Self-supervision from large unlabeled corpus of text: *slide* over an **anchor word** and its **context**:

.center[
<img src="./images/words.svg" style="width: 500px;" />
]

.footnote.small[
Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." NIPS 2013
]

---

# Word2Vec: CBoW

CBoW: representing the context as **Continuous Bag-of-Words**

Self-supervision from large unlabeled corpus of text: *slide* over an **anchor word** and its **context**:

.center[
<img src="./images/word2vec_words.svg" style="width: 500px;" />
]

.footnote.small[
Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." NIPS 2013
]

Question: dim of output embedding vs dim of input embedding

---
# Word2Vec: Skip Gram

.center[
<img src="./images/word2vec_skipgram.svg" style="width: 500px;" />
]

- Given the central word, predict occurence of other words in its context.
- Widely used in practice

---

# Word2Vec: Negative Sampling

- Task is simplified further: **binary classification** of word pairs
- For the sentence "The quick brown fox jumps over the lazy dog":
  - "quick" and "fox" are positive examples (if context window is 2)
  - "quick" and "apple" are negative examples
- By sampling negative examples, we don't just bring similar words' embeddings closer, but also push away dissimilar words' embeddings.

---

# Transformer-based methods

- **Attention** mechanism: more recent and more powerful than Word2Vec
- **BERT** (Bidirectional Encoder Representations from Transformers) allows for **contextual embeddings** (different embeddings for the same word in different contexts)
- For example, "bank" in "river bank" and "bank account" will have different embeddings
- This means converting a word to a vector is no longer a simple lookup in a table, but a function of the entire sentence

---

# Transformer-based methods

- **Sub-word tokenization**: BERT uses a sub-word tokenization, which allows it to handle out-of-vocabulary words better than Word2Vec
- For example, "unbelievable" can be split into "un" and "believable"
- This means that the model can guess the meaning of words it has never seen before, based on the meanings of their parts
- OpenAI tokenization example: [https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer)

---

# Take Away on Embeddings

**For text applications, inputs of Neural Networks are Embeddings**

---

# Take Away on Embeddings

- If **little training data** and a wide vocabulary not well covered by training data, use **pre-trained self-supervised embeddings** (word2vec, or with more time and resources, BERT, GPT, etc.)
- If **large training data** with labels, directly learn task-specific embedding for more precise representation.
- word2vec uses **Bag-of-Words** (BoW): they **ignore the order** in word sequences
- Depth &amp; non-linear activations on hidden layers are not that useful for BoW text classification.

---

# Language Modelling and Recurrent Neural Networks

---

## Language Models

Assign a probability to a sequence of words, such that plausible sequences have
higher probabilities e.g:

- $p(\text{"I like cats"}) > p(\text{"I table cats"})$
- $p(\text{"I like cats"}) > p(\text{"like I cats"})$

---

Likelihoods are factorized:

$p\_{\theta}(w\_{0})$

$p_{\theta}$ is parametrized by a neural network.

---

## Language Models

Assign a probability to a sequence of words, such that plausible sequences have
higher probabilities e.g:

- $p(\text{"I like cats"}) > p(\text{"I table cats"})$
- $p(\text{"I like cats"}) > p(\text{"like I cats"})$

Likelihoods are factorized:

$p\_{\theta}(w\_{0}) \cdot p\_{\theta}(w\_{1} | w\_{0})$

$p_{\theta}$ is parametrized by a neural network.

---

## Language Models

Assign a probability to a sequence of words, such that plausible sequences have
higher probabilities e.g:

- $p(\text{"I like cats"}) > p(\text{"I table cats"})$
- $p(\text{"I like cats"}) > p(\text{"like I cats"})$

Likelihoods are factorized:

$p\_{\theta}(w\_{0}) \cdot p\_{\theta}(w\_{1} | w\_{0}) \cdot \ldots \cdot p\_{\theta}(w\_n | w\_{n-1}, w\_{n-2}, \ldots, w\_0)$

$p_{\theta}$ is parametrized by a neural network.

---

The internal representation of the model can better capture the meaning
of a sequence than a simple Bag-of-Words.

---
## Conditional Language Models

NLP problems expressed as **Conditional Language Models**:

**Translation:** $p(Target | Source)$
- *Source*: "J'aime les chats"
- *Target*: "I like cats"

---

Model the output word by word:

$p\_{\theta}(w\_{0} | Source)$

---

## Conditional Language Models

NLP problems expressed as **Conditional Language Models**:

**Translation:** $p(Target | Source)$
- *Source*: "J'aime les chats"
- *Target*: "I like cats"

Model the output word by word:

$p\_{\theta}(w\_{0} | Source) \cdot p\_{\theta}(w\_{1} | w\_{0}, Source) \cdot \ldots $

---

## Conditional Language Models

**Question Answering / Dialogue:**

$p( Answer | Question , Context )$

- *Context*:
  - "John puts two glasses on the table."
  - "Bob adds two more glasses."
  - "Bob leaves the kitchen to play baseball in the garden."
- *Question*: "How many glasses are there?"
- *Answer*: "There are four glasses."

---

**Image Captionning:** $p( Caption | Image )$

- Image is usually the $2048$-d representation from a CNN

Question: do you have any idea of those NLP tasks that could be tackled
with a similar conditional modeling approach?

---

## Simple Language Model

.center[
<img src="./images/fixedsize_mlp.svg" style="width: 400px;" />
]

---

## Fixed context size

- **Average embeddings**: (same as CBoW) no sequence information
- **Concatenate embeddings**: introduces many parameters
- Still does not take well into account varying sequence sizes and sequence dependencies

Question: What's the dimension of concatenate embeddings?

---

## Recurrent Neural Network

.center[
<img src="./images/rnn_simple.svg" style="width: 200px;" />
]

---

Unroll over a sequence $(x_0, x_1, x_2)$:

.center[
<img src="./images/unrolled_rnn_3.svg" style="width: 400px;" />
]
---

## Recurrent Neural Network

.center[
<img src="./images/rnn_simple.svg" style="width: 200px;" />
]

Unroll over a sequence $(x_0, x_1, x_2)$:

.center[
<img src="./images/unrolled_rnn_2.svg" style="width: 400px;" />
]
---

## Recurrent Neural Network

.center[
<img src="./images/rnn_simple.svg" style="width: 200px;" />
]

Unroll over a sequence $(x_0, x_1, x_2)$:

.center[
<img src="./images/unrolled_rnn.svg" style="width: 400px;" />
]

---

## Language Modelling

.center[
<img src="./images/unrolled_rnn_words.svg" style="width: 450px;" />
]

**input** $(w\_0, w\_1, ..., w\_t)$ .small[ sequence of words ( 1-hot encoded ) ] <br/>
**output** $(w\_1, w\_2, ..., w\_{t+1})$ .small[shifted sequence of words ( 1-hot encoded ) ]

---

## Language Modelling

.center[
<img src="./images/unrolled_rnn_words.svg" style="width: 450px;" />
]

$x\_t = \text{Emb}(w\_t) = \mathbf{E} w\_t$ .right[input projection .red[`H`]]

---

$h\_t = g(\mathbf{W^h} h\_{t-1} + x\_t + b^h)$ .right[recurrent connection .red[`H`]]

---

$y = \text{softmax}( \mathbf{W^o} h\_t + b^o )$ .right[output projection .red[`K = |V|`]]

---

## Recurrent Neural Network

.center[
<img src="./images/unrolled_rnn_words.svg" style="width: 450px;" />
]

Input embedding $\mathbf{E}$  .right[.red[`|V| x H`]]

---

Recurrent weights $\mathbf{W^h}$  .right[.red[`H x H`]]

---

Output weights $\mathbf{W^{out}}$ .right[ .red[`H x K = H x |V|`]]

---

## Backpropagation through time

Similar as standard backpropagation on unrolled network

.center[
<img src="./images/unrolled_rnn_backwards_1.svg" style="width: 400px;" />
]

---

## Backpropagation through time

Similar as standard backpropagation on unrolled network

.center[
<img src="./images/unrolled_rnn_backwards_2.svg" style="width: 400px;" />
]

---

## Backpropagation through time

Similar as standard backpropagation on unrolled network

.center[
<img src="./images/unrolled_rnn_backwards_3.svg" style="width: 400px;" />
]

<br/>

---

- Similar as training **very deep networks** with tied parameters
- Example between $x_0$ and $y_2$: $W^h$ is used twice
- Usually truncate the backprop after $T$ timesteps
- Difficulties to train long-term dependencies

---

## Other uses: Sentiment Analysis

.center[
<img src="./images/unrolled_rnn_one_output_2.svg" style="width: 600px;" />
]

- Output is sentiment (1 for positive, 0 for negative)
- Very dependent on words order
- Very flexible network architectures

---

## Other uses: Sentiment analysis

.center[
<img src="./images/unrolled_rnn_one_output.svg" style="width: 600px;" />
]

- Output is sentiment (1 for positive, 0 for negative)
- Very dependent on words order
- Very flexible network architectures

---

# LSTM
.center[
<img src="./images/unrolled_lstm_2.svg" style="width: 500px;" />
]

.footnote.small[
Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 1997
]

---

# LSTM
.center[
<img src="./images/unrolled_lstm_1.svg" style="width: 500px;" />
]

.footnote.small[
Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 1997
]

---

# LSTM
.center[
<img src="./images/unrolled_lstm.svg" style="width: 500px;" />
]

.footnote.small[
Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 1997
]

---

- 4 times more parameters than RNN
- Mitigates **vanishing gradient** problem through **gating**
- Widely used and SOTA in many sequence learning problems

---

## Vanishing / Exploding Gradients

Passing through $t$ time-steps, the resulting gradient is the **product** of many gradients and activations.

- Gradient messages close to $0$ can shrink be $0$
- Gradient messages larger than $1$ can explode
- **LSTM** mitigates that in RNNs
- **Additive path** between $c\_t$ and $c\_{t-1}$
- **Gradient clipping** prevents gradient explosion
- Well chosen **activation function** is critical (tanh)

**Skip connections** in ResNet also alleviate a similar optimization problem.

---

# Next: Lab 6!