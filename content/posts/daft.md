---
date: '2026-03-30T22:14:01+08:00'
draft: false
title: 'Distribution Aware Fine-Tuning'
math: true
---

## Background

Let $\pi_\theta(y \mid x)$ be the model distribution, and let $\mathcal{D}$ be the input distribution. For each prompt $x \sim \mathcal{D}$, the model generates an output $y \sim \pi_\theta(\cdot \mid x)$. We are interested in maximizing the expected reward:
$$J(\theta) := \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}[r(x,y)]$$
where $r(x,y)$ measures the quality of the response, for example via correctness, BLEU score, or human feedback.

In an offline setting, however, we usually do **not** sample fresh responses from $\pi_\theta$ during every update. Instead, we observe pairs $(x,y)$ collected from some reference or data-collection distribution, which we denote by $\pi_{\mathrm{ref}}(y \mid x)$. If $\pi_{\mathrm{ref}}$ has support wherever $\pi_\theta$ does, then importance sampling gives:
$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\mathrm{ref}}(\cdot \mid x)} \left[r(x,y) \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}\right]$$

This identity is exact, but only when $\pi_{\mathrm{ref}}(y \mid x)$ is the actual sampling distribution and is known on the support of interest.

In practice, a supervised dataset often contains only one response $y_x$ for each $x$. In this case, we may write the empirical objective as:
$$\widehat{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[r(x,y_x) \frac{\pi_\theta(y_x \mid x)}{\pi_{\mathrm{ref}}(y_x \mid x)}\right]$$
with the important caveat that $\pi_{\mathrm{ref}}(y_x \mid x)$ is usually unknown. This missing denominator is the central difficulty in distribution-aware fine-tuning.

## Special Cases

To simplify the discussion, consider the setting in which the observed response $y_x$ is perfectly correct, so $r(x,y_x)=1$. This is the standard assumption behind supervised fine-tuning (SFT). Then the empirical importance-weighted objective becomes:
$$\widehat{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[\frac{\pi_\theta(y_x \mid x)}{\pi_{\mathrm{ref}}(y_x \mid x)}\right]$$

The issue remains that $\pi_{\mathrm{ref}}(y_x \mid x)$ is typically unavailable. Below are two useful surrogate choices to bypass this.

### Case 1: Point-mass surrogate, $\pi_{\mathrm{ref}}(y_x \mid x) \approx 1$

The simplest surrogate is to replace the unknown denominator with 1:
$$\widehat{J}_{\mathrm{DFT}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[\pi_\theta(y_x \mid x)\right]$$

Strictly speaking, this does **not** imply that the true reference distribution is uniform. Rather, it is a surrogate that intentionally ignores the unknown normalization term. Its gradient is:
$$\nabla_\theta \widehat{J}_{\mathrm{DFT}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[\pi_\theta(y_x \mid x)\nabla_\theta \log \pi_\theta(y_x \mid x)\right]$$

This objective optimizes probability directly in the **probability space**, not the log-probability space used by standard SFT. Therefore, it is generally **not** equivalent to maximum likelihood, although it shares the same maximizer in an idealized, noiseless one-hot setting. This objective is the most closely related to DFT-style training.

### Case 2: Self-normalized surrogate, $\pi_{\mathrm{ref}}(y_x \mid x) \approx sg(\pi_\theta(y_x \mid x))$

Another approach is to assume that the data sampling is on-policy, so the unknown denominator is approximated by the current model distribution. This yields the surrogate objective:
$$\widehat{J}_{\mathrm{sg}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[\frac{\pi_\theta(y_x \mid x)}{sg(\pi_\theta(y_x \mid x))}\right]$$
where $sg(\cdot)$ denotes the stop-gradient operator.

The objective value above is identically 1 for each sample, but its gradient is non-trivial because the denominator is treated as a constant during backpropagation:
$$\nabla_\theta \frac{\pi_\theta(y_x \mid x)}{sg(\pi_\theta(y_x \mid x))} = \frac{\nabla_\theta \pi_\theta(y_x \mid x)}{sg(\pi_\theta(y_x \mid x))} = \nabla_\theta \log \pi_\theta(y_x \mid x)$$

Consequently, this surrogate yields exactly the same gradient as the standard SFT objective $\mathbb{E}[\log \pi_\theta(y_x \mid x)]$. More precisely, it is a **gradient-equivalent surrogate** for SFT, rather than an equality of objective values.

## Estimation of the Reference Distribution

The previous section demonstrates that the denominator is the key quantity, yet in ordinary supervised data, it remains unknown. A natural solution is to introduce an auxiliary **proposal distribution** that is easily computable.

In summary, we need to find a distribution $\widetilde{\pi}_{\mathrm{prop}}(\cdot \mid x)$ such that:
* The newly sampled output $\widetilde{y}_x$ remains correct.
* The probability mass $\widetilde{\pi}_{\mathrm{prop}}(\widetilde{y}_x \mid x)$ is easy to evaluate.

One effective construction utilizes self-distillation or self-refinement. Given $x$ and the original ground-truth answer $y_x$, let a **frozen** teacher model $\pi_\phi$ produce a refined answer $\widetilde{y}_x$ from the conditional distribution:
$$\widetilde{\pi}_{\mathrm{prop}}(\cdot \mid x, y_x) := \pi_\phi(\cdot \mid x, y_x)$$

Here, $\phi$ must remain fixed (e.g., an earlier checkpoint or a separate, larger teacher model). This point is critical: if the proposal distribution changes with the current optimization variable $\theta$, the estimator and its gradient become significantly harder to interpret and stabilize.

If we sample:
$$\widetilde{y}_x \sim \widetilde{\pi}_{\mathrm{prop}}(\cdot \mid x, y_x)$$
we can define a new, computable surrogate objective:
$$\widehat{J}_{\mathrm{prop}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, \widetilde{y}_x \sim \widetilde{\pi}_{\mathrm{prop}}(\cdot \mid x, y_x)} \left[ r(x,\widetilde{y}_x) \frac{\pi_\theta(\widetilde{y}_x \mid x)}{\widetilde{\pi}_{\mathrm{prop}}(\widetilde{y}_x \mid x, y_x)} \right]$$

This construction is theoretically valid **as an importance-weighted objective with respect to the proposal distribution $\widetilde{\pi}_{\mathrm{prop}}$**. However, it is crucial to understand what this formulation does and does not provide:

* It **does** provide a computable denominator, because $\widetilde{\pi}_{\mathrm{prop}}(\widetilde{y}_x \mid x, y_x)$ can be directly evaluated by the frozen teacher via forward passes.
* It **does not** recover the original, unknown data-collection distribution $\pi_{\mathrm{ref}}(y \mid x)$. Instead, it actively defines a new, mathematically sound proposal distribution.
* The importance ratio is valid only when the support condition holds: whenever $\pi_\theta(\widetilde{y}_x \mid x) > 0$, we strictly require $\widetilde{\pi}_{\mathrm{prop}}(\widetilde{y}_x \mid x, y_x) > 0$.

## Intuition

The proposal distribution serves as a heuristic bridge to define a surrogate reference distribution. The underlying hope is that this proposal distribution will naturally align more closely with the model distribution than the original data-collection distribution did. If this holds true, the importance weights become more stable, and the resulting gradient estimates will exhibit lower variance. In particular, if the proposal distribution closely mirrors the model distribution, the importance weights will cluster around 1, bounding the surrogate objective close to the true expected reward.

Furthermore, by conditioning explicitly on the original response $y_x$, the proposal distribution leverages the structural and factual information embedded in the original dataset while still permitting exploration and refinement. A capable teacher model provides an informed proposal space that captures both task structure and response quality, ultimately driving better sample efficiency and faster convergence.

## Prompt Design

One practical prompt for sampling from the proposal distribution is:

```yaml
user_prompt: |
    This is a demonstration for a response to the following question:
    {x}

    The original response is:
    {y_x}

    Now, provide a response of your own. Ensure your reasoning is rigorous and your final conclusion is consistent with the demonstration. Use your own logical flow and natural expression.
```

This prompt should be viewed as a heuristic mechanism to instantiate $\widetilde{\pi}_{\mathrm{prop}}(\cdot \mid x, y_x)$, rather than a component of the theoretical guarantee itself. The mathematics only require that the sampling distribution is well-defined and that its probability mass on the sampled outputs is accurately computable.

## Empirical Results

Empirical results will be added in a future revision. Key questions to investigate include whether the proposal distribution noticeably improves sample efficiency, how sensitive the convergence is to variations in prompt design, and whether the resulting importance weights remain numerically stable across training steps.
