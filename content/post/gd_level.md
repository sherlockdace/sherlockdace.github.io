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

Before we start a formal discussion, we note that most first-order methods are iteratively defined in the following form:
$$\theta_{k+1} = \theta_k - \alpha p_k, \tag{1.1}$$
where $\alpha$ is the learning rate, and $p_k$ is the correction term added into the current term $\theta_k$.
Obviously, many methods we used, such as [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) and [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), can be viewed as special cases of equation (1.1). 
