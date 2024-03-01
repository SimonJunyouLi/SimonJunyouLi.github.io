---
title: Attention in the age of ADHD
subtitle: I explain what is attention and self-attention in short.
authors: Simon Li
layout: default
date: 2024-02-29
keywords: machine learning, transforemers
tags: ml
published: true
---

When I first got into machine learning, I always heard Transformer this and attention that. Unfortunately and frustratingly, I had absolute *ZERO* idea what attention was even after some brief search on the internet. Therefore, I will try to explain the concept of attention in a way that hopefully would be straight forward even for people who's new to machine learning or computer science.

## Why attention?

Before the arrival of Transformers, Recurrent Neural Network (RNN) was one of the most popular architecture for sequential data. RNN is known for its ability to generate local temporal dependcies However, it lacks the ability for long-range time dependencies, meaning that it would favor more recent information than further ones. To battle this weakness of leveraging information from hidden layers of RRN, attention mechanism is proposed. 

## What's Attention?

In short, the attention's output sequence is the weighted average of the input sequence.  More specifically, attention is a function that transforms an input sequence to an output sequence that does not necessary have the same length using a learned input-dependent weighted average. 

{% katexmm %}
$$ A: \text{ tokens } \to_{\text{weighted avg.}} \text{ tokens}$$
{% endkatexmm %}


### Math behind attention

In this section, we are going to formalize how the weighted average is taken for the output sequence from the input sequence. 

Suppose we have {% katexmm %}$T_{in}${% endkatexmm %} and {% katexmm %}$T_{out}${% endkatexmm %} number of input and output tokens, respectively. Then, we have an input sequence $V$ and an output sequence $Z$.

{% katexmm %}$$V \in \R^{T_{in}\times D} \quad Z \in \R^{T_{out} \times D} $${% endkatexmm %}

Let {% katexmm %}$p_{i,j}${% endkatexmm %} be the weight of input token {% katexmm %}$i${% endkatexmm %} in the output token {% katexmm %}$j${% endkatexmm %}. Then, we have that the output tokens are

{% katexmm %}$$ z_i = \sum^{T_{in}}_{j=1} p_{i,j}v_j \implies Z = PV$${% endkatexmm %}

Note that we'd require weighting coefficients {% katexmm %}$P \in [0,1]^{T_{out} \times T_{in}}${% endkatexmm %} and that {% katexmm %}$\sum^{T_{in}}_{j=1} p_{i,j} = 1${% endkatexmm %}.

### More math behind attention

We now know how the weighted average output is calculated. But where does the weighting coefficients {% katexmm %}$P${% endkatexmm %} come from? 

Suppose that we are given ***Query Tokens*** {% katexmm %}$Q \in \R^{T_{out}\times D_K}${% endkatexmm %} and ***Key tokens*** {% katexmm %}$K \in \R^{T_{in}\times D_K}${% endkatexmm %}. Then, we can determine the weight coefficient {% katexmm %}$p_{i,j}${% endkatexmm %} by calculating how similar {% katexmm %}$q_i${% endkatexmm %} and {% katexmm %}$k_j${% endkatexmm %} are. We normally would use cosine similarity to calculate the similarity; however, using just the numerator of cosine similarity, which is a dot product, not only works well but also saves a heafty amount of computations. 

After we obtained the raw similarity by using the inner product, we would scale the result by dividng it with {% katexmm %}$\sqrt{D_k}${% endkatexmm %} where {% katexmm %}$D_K${% endkatexmm %} is a scaling factor. This step is necessary as due to random initialization, we could have a sharp distribution of weight coefficients {% katexmm %}$P${% endkatexmm %}, which could take the model much more time to adjust the initial peaks. With the scaling factor, it ensures that the distribution at the start is more uniform, thus guaranteeing a faster convergence. 

Lastly, we normalize the value after scaling with softmax to obtain a probability distribution. 

{% katexmm %}$$P = \text{softmax} \left(\frac{QK^T}{\sqrt{D_K}}\right)$${% endkatexmm %}

## ***Self***-Sttention

Self-attention is a special case of attention where {% katexmm %}$T \coloneqq T_{in} = T_{out}${% endkatexmm %} and that all of {% katexmm %}$V, K, Q${% endkatexmm %} are derived from the **same** input token sequence {% katexmm %}$X \in \R^{T\times D}${% endkatexmm %}. This means that to calculate {% katexmm %}$V, K, Q${% endkatexmm %}, we have learnable parameters {% katexmm %}$W_V, W_K, W_Q${% endkatexmm %} such that 

{% katexmm %}$$\begin{aligned}
V &= XW_V \in \R^{T\times D}, W_V \in \R^{D \times D} \\
K &= XW_K \in \R^{T\times D_K}, W_k \in \R^{D \times D_K} \\
Q &= XW_Q \in \R^{T\times D_K}, W_Q \in \R^{D \times D_K}
\end{aligned}$${% endkatexmm %}

where {% katexmm %}$D_K${% endkatexmm %} is the dimension of the Keys and Queries tokens.

Therefore, we have that the output of the self-attention would be 

{% katexmm %}$$ Z = \text{softmax} \left( \frac{X W_Q W_K^T X^T}{\sqrt{D_K}} XW_V \right) $${% endkatexmm %}

## ***Multi-Head*** Self-Attention

Just like a convolution layer where we can run multiple convolutions, we can run **mutiple** attention heads per layer! Suppose the output of each head $h$ is given by $Z$ described above. Then, the final output is obtained by concatenating all the individual head's output and apply a linear transformation
{% katexmm %}$$Z = [Z_1, \dots, Z_H]W_O$${% endkatexmm %}
where {% katexmm %}$W_O \in \R^{HD_V\times D}${% endkatexmm %} is a learnable parameter. 