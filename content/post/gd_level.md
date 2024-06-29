---
title: "The Magnitude Level of First-Order Methods"
date: 2024-06-29T20:08:01+08:00
draft: false
math: true
---

In broad terms, the majority of training problems we encounter involve the task of locating a global or local minimum of a given function $f(\theta)$.
Within the field of deep learning, $\theta$ represents the parameter values to be determined for a given neural network structure.
Numerous approaches, particularly first-order methods, have been devised to identify the local or global optimal minimum point $\theta^*$.
In this blog, we aim to delve into the discussion surrounding the magnitude level of the correction term in these first-order methods.

## Some Classical First-Order Methods

Before we delve into a formal discussion, it's worth noting that many first-order optimization methods can be expressed iteratively using the following form:
$$\theta_{k+1} = \theta_k - \alpha p_k. \tag{1.1} $$
In this equation, $\theta_k$ represents the current parameter, $\alpha$ is the learning rate, and $p_k$ denotes the correction term added to the current parameter $\theta_k$.

Notably, widely used optimization methods such as [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) and [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) can be viewed as special cases of equation (1.1).
In SGD, the correction term $p_k$ is equivalent to the gradient term $\nabla_\theta f(\theta)$, where $\nabla_\theta$ represents the gradient with respect to the parameters $\theta$, and $f(\theta)$ denotes the objective function being optimized.
In contrast, Adam utilizes a different form of the correction term. Specifically, Adam computes the correction term as $m_k / \sqrt{v_k}$, where $m_k$ denotes the first momentum term and $v_k$ represents the second momentum term.
