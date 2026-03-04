---
title: "Reinforcement Learning Algorithms and Importance Sampling"
date: 2026-02-05T21:33:55+08:00
draft: false
math: true
---

# Taming Importance Sampling in RL: A Divergence-Constrained Approach

If you've spent any time working on off-policy Reinforcement Learning (RL), you're intimately familiar with **Importance Sampling (IS)**. It's the standard trick used to estimate the expected return of a new target policy $\pi_\theta$ using samples collected from an older behavior policy $\pi_b$. 

However, as many practitioners know, IS can be notoriously unstable. In this post, we'll dive into the math behind *why* it fails—specifically looking at the measure-theoretic misalignment of distribution supports—and introduce a theoretical framework called **Reinforcement Learning with Constraints (RLC)** to fix it.

---

## 1. The Problem: Misalignment of Distribution Supports

Let's start from the top. In RL, our goal is to maximize the expected reward under the stationary state occupancy distribution induced by our policy. For notational simplicity, let us denote this occupancy distribution as $\pi_\theta(s)$. The objective is:

$$ \max_{\theta} \mathcal{J}(\theta) := \mathbb{E}_{s \sim \pi_{\theta}} [r(s)] = \int_{\mathcal{S}} r(s) \, \pi_{\theta}(s) \, ds $$

Since we cannot always sample directly from $\pi_{\theta}$, we use samples from the behavior distribution $\pi_b$ via Importance Sampling. Assuming absolute continuity ($\pi_{\theta} \ll \pi_b$), we rewrite this as:

$$ \mathcal{J}(\theta) = \mathbb{E}_{s \sim \pi_b} \left[ r(s) \cdot \underbrace{\frac{\pi_{\theta}(s)}{\pi_b(s)}}_{w_\theta(s)} \right] $$

This works fine in theory, but the importance weight $w_{\theta}(s)$ can suffer from massive variance. Practitioners often patch this with weight clipping or KL penalties, but there is a deeper, fundamental issue: **What if the target distribution explores regions of the state space that the behavior distribution completely ignored?**

If $\text{supp}(\pi_{\theta}) \not\subseteq \text{supp}(\pi_b)$, the importance weights blow up to infinity. To formalize this, we invoke the **Lebesgue Decomposition Theorem**. We can decompose the target distribution $\pi_\theta$ into an absolutely continuous part ($\mu_{ac} \ll \pi_b$) and a singular part ($\mu_s \perp \pi_b$). 

This splits our objective into two components:
$$ \mathcal{J}(\theta) = \underbrace{\int_{\mathcal{A}} r(s) \, \mu_{ac}(s)}_{\mathcal{J}_{ac}(\theta)} + \underbrace{\int_{\mathcal{B}} r(s) \, \mu_{s}(s)}_{\mathcal{J}_{s}(\theta)} $$
where $\mathcal{A}$ is the overlapping region, and $\mathcal{B}$ is the exclusive region where only $\pi_\theta$ has support.

This reveals two massive headaches for standard IS:
1.  **Variance in $\mathcal{A}$:** Even where they overlap, if $\pi_b(s)$ is tiny compared to $\pi_\theta(s)$, the weights explode.
2.  **Bias from $\mathcal{B}$:** Because $\pi_b$ never samples from $\mathcal{B}$, standard IS completely ignores $\mathcal{J}_s(\theta)$, creating an irreducible bias.

---

## 2. Reinforcement Learning with Constraints (RLC)

Instead of applying post-hoc fixes like clipping, what if we proactively constrain the optimization? We can force the target policy to stay close to the behavior policy so that the support mismatch never gets out of hand.

We can formulate this as a constrained optimization problem using a generic $f$-divergence $D_f(\pi_\theta \| \pi_b)$:
$$
\begin{aligned}
\max_{\theta} \quad & \mathbb{E}_{s \sim \pi_{\mathrm{b}}}\left[ w_\theta(s) \cdot r(s)\right] \\
\text{s.t.} \quad & D_f(\pi_\theta \| \pi_{\mathrm{b}}) \leq \delta
\end{aligned}
$$

Solving this directly is difficult. However, we can use **Lagrangian relaxation** to turn it into an unconstrained regularized objective. Under strong duality conditions, any local optimum of the constrained problem is a stationary point of:
$$ \max_{\theta} \quad \mathcal{J}_{\mathrm{reg}}(\theta) := \mathcal{J}(\theta) - \lambda D_f(\pi_\theta \| \pi_{\mathrm{b}}) $$
where $\lambda \ge 0$ is the Lagrange multiplier controlling the strength of the constraint.

---

## 3. Deriving the Surrogate Objective

Optimizing $\mathcal{J}_{\mathrm{reg}}$ directly can still be tricky because the divergence term depends on the density ratio. A modern approach is to derive a **surrogate loss** that matches the optimal policy condition. This transforms the problem into minimizing a Mean Squared Error (MSE) loss, which is far more tractable for gradient-based methods.

### The Optimal Policy Condition

First, we characterize what the optimal policy $\pi_\theta^*$ looks like. Since $\pi_\theta$ must be a valid probability distribution, it must satisfy the normalization constraint $\int_{\mathcal{S}} \pi_\theta(s) \, ds = 1$. We introduce a Lagrange multiplier $\eta \in \mathbb{R}$ for this constraint. 

The Lagrangian over the function space of densities becomes:
$$
\mathcal{L}(\pi_\theta, \eta) = \int_{\mathcal{S}} r(s) \pi_\theta(s) \, ds - \lambda \int_{\mathcal{S}} f\left(\frac{\pi_\theta(s)}{\pi_{\mathrm{b}}(s)}\right) \pi_{\mathrm{b}}(s) \, ds - \eta \left( \int_{\mathcal{S}} \pi_\theta(s) \, ds - 1 \right)
$$

To find the optimal density, we take the functional derivative of $\mathcal{L}$ with respect to $\pi_\theta(s)$ and set it to zero:
$$
\frac{\delta \mathcal{L}}{\delta \pi_\theta(s)} = r(s) - \lambda f'\left(\frac{\pi_\theta(s)}{\pi_{\mathrm{b}}(s)}\right) - \eta = 0
$$

Let $C := \eta$ (a constant independent of $s$) and let $w^*(s) = \frac{\pi_\theta^*(s)}{\pi_{\mathrm{b}}(s)}$ be the optimal importance weight. We arrive at the characterizing condition for the optimal policy:
$$ r(s) - \lambda f'(w^*(s)) = C $$

This elegant result links the reward, the regularization coefficient, and the derivative of the $f$-divergence generating function directly to the optimal importance weight.

### Building the Surrogate Loss

Since the optimal condition implies $r(s) - C - \lambda f'(w^*(s)) = 0$, we can construct a surrogate learning objective by penalizing the squared difference from this condition. This gives us the MSE surrogate loss $\mathcal{L}_{\mathrm{sur}}$:

$$ \mathcal{L}_{\mathrm{sur}}(\theta) = \mathbb{E}_{s \sim \pi_{\mathrm{b}}} \left[ \left(r(s) - C - \lambda f'\left(w_\theta(s)\right) \right)^2 \right] $$

Expanding the quadratic term:
$$ \mathcal{L}_{\mathrm{sur}}(\theta) = \mathbb{E}_{s \sim \pi_{\mathrm{b}}} \left[ (r(s) - C)^2 - 2\lambda (r(s) - C) f'\left(w_\theta(s)\right) + \lambda^2 \left(f'\left(w_\theta(s)\right)\right)^2 \right] $$

Notice that the first term, $\mathbb{E}_{s \sim \pi_{\mathrm{b}}} \left[ (r(s) - C)^2 \right]$, does not depend on $\theta$. We can ignore it during optimization. Thus, **minimizing** the surrogate loss $\mathcal{L}_{\mathrm{sur}}$ is mathematically equivalent to **maximizing** the following surrogate objective:

$$ \mathcal{J}_{\mathrm{sur}}(\theta) = \mathbb{E}_{s \sim \pi_{\mathrm{b}}} \left[ (r(s) - C) f'\left(w_\theta(s)\right) - \frac{\lambda}{2} \left(f'\left(w_\theta(s)\right)\right)^2 \right] $$

---

## 4. Examples in Practice

By plugging in different $f$-divergences, we recover different practical algorithms. The constant $C$ acts as a baseline and is determined by enforcing the normalization condition $\mathbb{E}_{s \sim \pi_b}[w^*(s)] = 1$.

### 1. KL Divergence
*   **Generator:** $f(t) = t \log t - t + 1$
*   **Derivative:** $f'(t) = \log t$
*   **Optimal Weight:** $w^*(s) = \exp\left(\frac{r(s)-C}{\lambda}\right)$
*   **Constant $C$:** Solved via $\mathbb{E}[w^*(s)]=1 \Rightarrow C = \lambda \log \mathbb{E}_{\pi_b}[\exp(r(s)/\lambda)]$. (In practice, estimated via the log-sum-exp trick).
*   **Surrogate Objective:**
    $$ \mathcal{J}_{\mathrm{sur}}^{\mathrm{KL}}(\theta) = \mathbb{E}_{s \sim \pi_{\mathrm{b}}} \left[ (r(s) - C) \log w_\theta(s) - \frac{\lambda}{2} \left(\log w_\theta(s)\right)^2 \right] $$

### 2. $\chi^2$ Divergence
*   **Generator:** $f(t) = \frac{1}{2}(t-1)^2$
*   **Derivative:** $f'(t) = t - 1$
*   **Optimal Weight:** $w^*(s) = \frac{r(s)-C}{\lambda} + 1$
*   **Constant $C$:** Solved via $\mathbb{E}[w^*(s)]=1 \Rightarrow C = \mathbb{E}_{\pi_b}[r(s)]$.
*   **Surrogate Objective:**
    $$ \mathcal{J}_{\mathrm{sur}}^{\chi^2}(\theta) = \mathbb{E}_{s \sim \pi_{\mathrm{b}}} \left[ (r(s) - C) (w_\theta(s) - 1) - \frac{\lambda}{2} \left(w_\theta(s) - 1 \right)^2 \right] $$

### 3. Total Variation (TV) Distance
*   **Generator:** $f(t) = \frac{1}{2}|t - 1|$
*   **Derivative:** $f'(t) = \frac{1}{2}\mathrm{sgn}(t - 1)$ (subgradient at $t=1$)
*   **Constant $C$:** The condition $\mathbb{E}[w^*(s)] = 1$ under TV implies $C$ is approximately the median of the reward distribution: $C \approx \mathrm{Median}_{\pi_{\mathrm{b}}}(r(s))$.
*   **Surrogate Objective:**
    $$ \mathcal{J}_{\mathrm{sur}}^{\mathrm{TV}}(\theta) = \mathbb{E}_{s \sim \pi_{\mathrm{b}}} \left[ \frac{1}{2} (r(s) - C) \mathrm{sgn}\left(w_\theta(s) - 1\right) - \frac{\lambda}{8} \right] $$
    *(Note: Since $\mathrm{sgn}^2(x) = 1$ almost everywhere, the regularization term becomes a constant penalty.)*

### 4. Reverse KL Divergence
*   **Generator:** $f(t) = -\log t + t - 1$
*   **Derivative:** $f'(t) = 1 - 1/t$
*   **Optimal Weight:** $w^*(s) = \frac{\lambda}{\lambda + C - r(s)}$
*   **Constant $C$:** Root of the equation $\mathbb{E}_{\pi_b}\left[ \frac{\lambda}{\lambda + C - r(s)} \right] = 1$.
*   **Surrogate Objective:**
    $$ \mathcal{J}_{\mathrm{sur}}^{\mathrm{RKL}}(\theta) = \mathbb{E}_{s \sim \pi_{\mathrm{b}}} \left[ (r(s) - C) \left(1 - \frac{1}{w_\theta(s)}\right) - \frac{\lambda}{2} \left(1 - \frac{1}{w_\theta(s)}\right)^2 \right] $$

---

## 5. Conclusion

By choosing the right $f$-divergence, you can explicitly control the trade-off between bias, variance, and the shape of the desired target distribution. 

*   **KL** encourages smooth, exponential weighting (common in policy gradients).
*   **$\chi^2$** provides a linear relationship between reward and weight (robust to outliers).
*   **TV** focuses on the median reward and creates a hard threshold on weights.

This framework moves us away from hacking around with arbitrary weight clipping toward **principled, theoretically grounded surrogate objectives**. The constant $C$ can be treated as a learnable parameter (dual variable) updated via gradient ascent to satisfy the constraint, making this approach ready for modern deep RL pipelines.