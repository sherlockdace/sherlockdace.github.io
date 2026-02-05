---
title: "Reinforcement Learning Algorithms and Importance Sampling"
date: 2026-02-05T21:33:55+08:00
draft: true
math: true
---

In this blog, we will have a deep discussion of the importance sampling (IS) and why it is so important in reinforcement learning (RL), espically in the context of large language models (LLMs).

## Importance Sampling and Radon–Nikodym theorem

First we discuss the IS. For a given function $f(x)$, with $x$ is sampled from a probability $\mu$. Then the expectation is given by

$$ 
\mathcal{J} (\mu) := \mathbb{E}_{\mu} [f (x)] = \int f(x) d \mu(x). 
$$

To get an approximation of $\mathcal{J}$, we only need to sample $(x_j)_{j=1}^n$ from $\mu (x)$.  Thus, the numerical approximation is given by

$$
\mathcal{J}^{na} := \sum_j f(x_j) \mu (x_j). 
$$