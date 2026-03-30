---
date: '2026-03-30T22:14:01+08:00'
draft: true
title: 'Distribution Aware Fine-Tuning'
math: true
---

# Distribution Aware Fine-Tuning, DAFT

## Background

Given a LLM distribution $\pi_\theta$, and the data distribution $\mathcal{D}$. Let $x \sim \mathcal{D}$ be a data point, and $y \sim \pi_\theta(\cdot|x)$ be the model's output. We want to maximize the reward $r(x,y)$, which is a function of the data point and the model's output. The reward can be any function that measures the quality of the model's output, such as accuracy, BLEU score, or human feedback.

Formally, we consider the following maximization problem:
$$
\max_{\pi_\theta} \quad \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} ~ r(x,y)
$$

However, sampling $y$ from any distribution $\pi_\theta (\cdot | x)$ is expensive and impractical. Therefore, we employ the importance sampling (IS) technique to transfer the distribution from $\pi_\theta$ to a more tractable distribution $\pi_{ref}$. In particular, we can rewrite the maximization problem as follows:
$$
\max_{\pi_\theta} \quad \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{ref}(\cdot|x)} ~ r(x,y) \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

For simplicity to discuss the problem, we assume that only one $y_x$ is sampled for each $x$. In general, note that we may assume that $x$ is repeatedly sampled from $\mathcal{D}$. Thus, the above formulation is equivalent to sampling multiple $y$ from $\pi_\theta$ directly. 

For ease of notation, we can rewrite the above maximization problem as follows:
$$
\max_{\pi_\theta} \quad \mathbb{E}_{x \sim \mathcal{D}} ~ r(x,y_x) \frac{\pi_\theta(y_x|x)}{\pi_{ref}(y_x|x)} 
$$

## Special Cases

In general, the reward $r(x, y)$ can be either positive or negative. However, we consider the following setting: The reference distribution $\pi_{ref}$ is so perfect such that all the samples $y_x$ are correct for each $x$. In this case, we can set $r(x,y_x) = 1$ for all $x$ and $y_x$. Thus, the maximization problem is simplified as follows:
$$
\max_{\pi_\theta} \quad \mathbb{E}_{x \sim \mathcal{D}} ~ \frac{\pi_\theta(y_x|x)}{\pi_{ref}(y_x|x)} 
$$

Note that the "all correct" setting is natural, espically when considering the SFT setting. In SFT, we assume that the reference distribution $\pi_{ref}$ is the human teacher, which is perfect and always provides correct answers. Therefore, the reward is always 1 for all samples.

Though the problem looks simple, the reference distribution $\pi_{ref}$ is unaccessible in practice! The reason is easy: the origin of the data is unknown and from vast sources. Thus, we cannot know the exact distribution of the data. Therefore, we need to estimate the reference distribution $\pi_{ref}$ from the data. In parctice, there are two common assumptions for the reference distribution $\pi_{ref}$:

### Case 1: $\pi_{ref}(y_x | x) \equiv 1$

The simplest assumption is that the reference distribution $\pi_{ref}$ is uniform, i.e., $\pi_{ref}(y_x | x) \equiv 1$ for all $x$ and $y_x$. It means that the reference distribution is completely confident on the correct answer $y_x$ for each $x$. In this case, the maximization problem is simplified as follows:
$$
\max_{\pi_\theta} \quad \mathbb{E}_{x \sim \mathcal{D}} ~ \pi_\theta(y_x|x) 
$$
In this case, the problem degrades to the DFT algorithm, which is popular these days. 

### Case 2: $\pi_{ref}(y_x | x) \equiv \pi_\theta(y_x | x)$

Another common assumption is that the reference distribution $\pi_{ref}$ is the same as the model distribution $\pi_\theta$, i.e., $\pi_{ref}(y_x | x) \equiv sg(\pi_\theta(y_x | x))$ for all $x$ and $y_x$. Here, $sg(\cdot)$ denotes the stop-gradient operation which means that the reference distribution is treated as a constant and does not receive gradients during optimization. It means that the reference distribution is exactly the same as the model distribution. In this case, the maximization problem is simplified as follows:
$$
\max_{\pi_\theta} \quad \mathbb{E}_{x \sim \mathcal{D}} ~ \frac{\pi_\theta(y_x|x)}{sg(\pi_\theta(y_x|x))} 
$$
Taking the gradient of the above objective with respect to $\theta$, we have:
$$
\nabla_\theta \frac{\pi_\theta(y_x|x)}{sg(\pi_\theta(y_x|x))} = \frac{\pi_\theta (y_x | x) \nabla \log \pi_\theta(y_x|x)}{sg(\pi_\theta(y_x|x))} = \nabla \log \pi_\theta(y_x|x).
$$
It means that the gradient of the above objective is exactly the same as the gradient of the SFT objective. Therefore, the above maximization problem is equivalent to the SFT algorithm.

## Estimation of the Reference Distribution

As shown in the above two cases, the reference distribution $\pi_{ref}$ is unaccessible in many cases, and we have to make some assumptions to estimate the reference distribution $\pi_{ref}$. In this work, we try to provide the first systematic study on the estimation of the reference distribution $\pi_{ref}$. In particular, we do not estimate it, we calculate it! And the details are given as follows. 

Recenly, the self-distillation becomes popular again. Specifically, for a given reference distribution $\pi_{ref}$, we input the data point $x$ and the original response $y_x$ into the model, and force the model to output an answer by its own and get $\tilde{y}_x$. Since the model has known the truth answer $y_x$, we can assume that the output $\tilde{y}_x$ is always correct. 

We claim that we can get the unkown reference distribuiton now. In fact, we define the drifted reference distribution $\tilde{\pi}_{ref}$ as follows:
$$
\tilde{\pi}_{ref}( \cdot | x) := \pi_\theta( \cdot | x, y_x).
$$
Now, we can actually calcuate the refernce model now. Specifically, we have the following new maximization problem:
$$
\max_{\pi_\theta} \quad \mathbb{E}_{x \sim \mathcal{D}} ~ r(x,\tilde{y}_x) \frac{\pi_\theta(\tilde{y}_x|x)}{\tilde{\pi}_{ref}(\tilde{y}_x|x)} = \frac{\pi_\theta(\tilde{y}_x|x)}{\pi_{ref}(\tilde{y}_x|x, y_x)}.
$$

We guess we need to provide some exalanations for the above problem and why it works. 

* First, we assume that the refined answer $\tilde{y}_x$ is always correct. Thus, we can set $r(x, \tilde{y}_x) = 1$ for all $x$ and $\tilde{y}_x$.

* For drifted reference distribution $\tilde{\pi}_{ref}$, $\tilde{y}_x$ is actually sampled from $\tilde{\pi}_{ref}(\cdot | x)$ which making the the importance sampling technique valid.

* The distribution $\tilde{\pi}_{ref}$ can be calculated in fact either from the training engine or the inference engine. Thus, we can get the exact reference distribution $\tilde{\pi}_{ref}$ without any estimation error.


## Prompt Design

The prompt for getting the drifted reference distribution $\tilde{\pi}_{ref}$ is given as follows:
```yaml
user_prompt: |
    This is a demonstration for a response to the following question:
    {x}

    The original response is:
    {y_x}

    Now, provide a response of your own. Ensure your reasoning is rigorous and your final conclusion is consistent with the demonstration. Use your own logical flow and natural expression.
```

We are not sure of the optimal prompt design for getting the drifted reference distribution $\tilde{\pi}_{ref}$. The above prompt is just a preliminary design. We will try to provide more prompt designs in the future.

## Empirical Results

Going to provide some empirical results in the future.