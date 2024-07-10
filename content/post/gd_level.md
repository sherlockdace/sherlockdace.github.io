---
title: "The Magnitude Level of First-Order Methods"
date: 2024-06-29T20:08:01+08:00
draft: 
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
In SGD, the correction term $p_k$ corresponds to the gradient term $\nabla_\theta f(\theta)$. 
On the other hand, Adam employs a different form of the correction term. Specifically, Adam calculates the correction term as $m_k / \sqrt{\hat{v}_k + \epsilon}$, where $m_k$ represents the first momentum term and $\hat{v}_k$ denotes the second momentum term. The term $\sqrt{\hat{v}_k + \epsilon}$ acts as a normalization factor for the first momentum term.
According to the [work](https://arxiv.org/abs/2405.14578), it claims that the normalization term in Adam approximates the update with the correction term $p_k := \text{sign} ( \nabla f(\theta_k) )$. 
Another interesting algorithm mentioned is Normalized SGD (NSGD). In this algorithm, the correction term is chosen to be the normalization of $\nabla f(\theta_k)$ or $m_k$.

The experiments demonstrate that both Adam and Normalized SGD (NSGD) consistently outperform standard SGD. 
This suggests that normalization techniques can significantly enhance training results in various applications.
Both NSGD and Adam aim to achieve a constant magnitude for the correction term $p_k$. In other words, they seek to ensure that the magnitude of $p_k$ is on the order of $O(1)$. However, it is worth noting that the specific choice of magnitude can vary.

The natural question arises: 
**What is the optimal magnitude level for $p_k$? Is it on the order of $O(1)$ or $O((\nabla f)^p)$ for some fixed value of $p$**?