---
date: '2026-04-13T14:14:23+08:00'
draft: false
title: 'Compatibility-Gated Policy Optimization'
math: true
---

Policy optimization for language models is usually framed around a behavior policy: collect responses from an old policy, form the importance ratio against the current policy, and then clip that ratio to keep training stable. That recipe is effective, but it also hard-wires the training objective to a specific sampling distribution and to the engineering machinery required to track it exactly.

Compatibility-Gated Policy Optimization (CGPO) starts from a different question. Instead of asking whether the current policy has moved too far from the behavior policy, it asks whether a sample is still sufficiently compatible with the current policy itself. If the answer is yes, use the sample. If the answer is no, drop it for the current update.

The idea is simple, but the math has to be stated carefully. In particular:

1. CGPO is **not** an unbiased importance-sampling estimator of the original on-policy objective.
2. A hard gate induces a new accepted-sample distribution that can be characterized exactly.
3. Any claim that CGPO "must improve the original RL objective" requires extra assumptions and should not be stated for free.

## 1. From Ratio Clipping to Compatibility Gating

Let $x \sim \mathcal D$ be a prompt and let $\pi_\theta(\cdot \mid x)$ be the current policy. The ideal on-policy RL objective is

$$ J_{\mathrm{RL}}(\theta) := \mathbb E_{x \sim \mathcal D, \, y \sim \pi_\theta(\cdot \mid x)}[r(x,y)]. $$

Suppose training data are instead sampled from some conditional distribution $\mu(\cdot \mid x)$. When $\pi_\theta(\cdot \mid x)$ is absolutely continuous with respect to $\mu(\cdot \mid x)$ for $\mathcal D$-almost every $x$, one may rewrite the objective as

$$ J_{\mathrm{RL}}(\theta) = \mathbb E_{x \sim \mathcal D, \, y \sim \mu(\cdot \mid x)}\left[\frac{\pi_\theta(y \mid x)}{\mu(y \mid x)} r(x,y)\right]. $$

PPO-style methods then replace the raw ratio with a clipped version in order to control variance. Conceptually, this means that sample usefulness is judged relative to the data-collection distribution $\mu$.

CGPO changes the criterion. It asks:

> Does the current policy still assign enough probability mass to this sample for it to be a useful training signal now?

That shift is attractive in LLM RL pipelines where data may come from a replay buffer, rejection sampling, preference data, or a mixture of stale policy checkpoints. In those settings, exact ratio tracking can be cumbersome, while sample compatibility with the current policy is easy to evaluate from token log-probabilities.

## 2. Setup

For optimization, it is more precise to work with a scalar training signal $A(x,y)$ rather than raw reward. In practice, $A$ may be a reward-model score, a centered reward, an advantage estimate, or a preference-derived scalar. Assume we have triples

$$ (x,y,A) \sim \mathcal B. $$

Write the response as $y = (y_1,\dots,y_T)$, where $T := |y|$. Define token log-probabilities under the current policy:

$$ \ell_{\theta,i}(x,y) := \log \pi_\theta(y_i \mid x, y_{ <i }). $$

Then

$$ \log \pi_\theta(y \mid x) = \sum_{i=1}^T \ell_{\theta,i}(x,y). $$

The natural length-normalized compatibility score is the geometric mean token probability:

$$ c_\theta(x,y) := \exp\left(\frac{1}{T}\sum_{i=1}^T \ell_{\theta,i}(x,y)\right) = \pi_\theta(y \mid x)^{1/T}. $$

Equivalently, one may use the average token log-probability

$$ \bar \ell_\theta(x,y) := \frac{1}{T}\sum_{i=1}^T \ell_{\theta,i}(x,y), $$

because $c_\theta(x,y) \ge \tau$ if and only if $\bar \ell_\theta(x,y) \ge \log \tau$.

At token level, define

$$ c_{\theta,i}(x,y) := \pi_\theta(y_i \mid x, y_{ <i }). $$

The geometric-mean sequence score is a better default than the raw sequence probability $\pi_\theta(y \mid x)$, because the latter shrinks exponentially with length and is therefore not comparable across responses of different lengths.

## 3. Intuition: When Is a Sample "Close Enough" to On-Policy?

Before defining the hard gate formally, it is worth stating the intuition carefully. CGPO is trying to answer a practical question:

> When can an offline or stale sample $(x,y)$ still be treated as a useful approximation to an on-policy sample for the current policy $\pi_\theta$?

The first part of the answer is sample-level. If the current policy assigns very low probability to many tokens in $y$, then $y$ is not representative of what $\pi_\theta$ would now generate. Training strongly on such a sample can inject high-variance or semantically stale gradients. By contrast, if the average token log-probability is high, then the response lies in a region that the current policy still supports. This is exactly what the compatibility score is measuring.

That said, high compatibility for one sample is only a heuristic signal. It does **not** prove that the sample was literally drawn from $\pi_\theta$. A single response can have high likelihood under multiple policies, and provenance is not identifiable from one observation alone.

So the mathematically correct object is not an individual sample, but the **distribution of accepted samples**. Suppose data come from some source distribution $\mu(\cdot \mid x)$ and we keep only samples that pass the compatibility gate. Then the accepted distribution $\tilde \mu_\theta(\cdot \mid x)$ should be viewed as approximately on-policy only when two conditions hold:

1. the source distribution $\mu(\cdot \mid x)$ is already reasonably close to $\pi_\theta(\cdot \mid x)$;
2. the gate removes mainly low-compatibility tail mass, so the acceptance rate $Z_\theta(x)$ remains high.

This intuition is formalized later by the bound

$$ \mathrm{TV}(\tilde \mu_\theta(\cdot \mid x), \pi_\theta(\cdot \mid x)) \le 1 - Z_\theta(x) + \mathrm{TV}(\mu(\cdot \mid x), \pi_\theta(\cdot \mid x)). $$

The bound says exactly what one would hope: accepted samples can be treated as approximately on-policy when the original data are not too stale and the gate is not excessively aggressive. In other words, CGPO does not claim that a high-compatibility sample is "really from $\pi_\theta$." It claims something weaker and more defensible: after filtering, the retained sample distribution can move closer to the current policy, and the remaining mismatch is controlled by source-policy drift plus discarded probability mass.

This perspective also explains why token-level gating is useful. A sequence may look acceptable on average while still containing a few locally implausible tokens. Sequence-level compatibility answers "is this response globally plausible under $\pi_\theta$?"; token-level compatibility answers "which parts of this response still look on-manifold under $\pi_\theta$?" The two gates therefore solve different approximation problems.

With that interpretation in place, we can define the objective cleanly.

## 4. Hard Gates and the CGPO Objective

CGPO introduces a gate based on current-policy compatibility. For a sequence-level threshold, define

$$ \tau(A) := \begin{cases} \tau_+, & A \ge 0, \\ \tau_-, & A < 0, \end{cases} \qquad 0 < \tau_+ \le \tau_- \le 1. $$

If $A = 0$, the sample contributes zero to the objective anyway, so the choice of threshold is immaterial.

The sequence-level hard gate is

$$ g_\theta(x,y,A) := \mathbf 1\!\left(c_\theta(x,y) \ge \tau(A)\right). $$

The basic sequence-level CGPO objective is

$$ \mathcal L_{\mathrm{seq}}(\theta) := \mathbb E_{(x,y,A) \sim \mathcal B}\left[A(x,y) \, g_\theta(x,y,A) \, \log \pi_\theta(y \mid x)\right]. $$

This already captures the core idea: a sample is updated only if the current policy considers it sufficiently plausible.

For finer control, introduce token-level thresholds

$$ \tau_i(A) := \begin{cases} \tau_{+,i}, & A \ge 0, \\ \tau_{-,i}, & A < 0, \end{cases} \qquad 0 < \tau_{+,i} \le \tau_{-,i} \le 1, $$

and token gates

$$ g_{\theta,i}(x,y,A) := \mathbf 1\!\left(c_{\theta,i}(x,y) \ge \tau_i(A)\right). $$

The joint sequence-and-token objective is then

$$ \mathcal L_{\mathrm{tok}}(\theta) := \mathbb E_{(x,y,A) \sim \mathcal B}\left[A(x,y) \, g_\theta(x,y,A) \sum_{i=1}^T g_{\theta,i}(x,y,A) \, \ell_{\theta,i}(x,y)\right]. $$

This two-level design has a clear interpretation:

1. The sequence gate decides whether the response is globally compatible with the current policy.
2. The token gates remove locally implausible tokens even when the overall sequence passes.

The asymmetry $\tau_+ < \tau_-$ is intentional. Positive samples may admit many acceptable realizations, so the gate should be permissive. Negative samples are most useful when they are still likely mistakes under the current policy, so the gate should be stricter.

## 5. What Distribution Does the Gate Induce?

The cleanest way to analyze CGPO is to treat the gate as a selection rule applied to samples from a data distribution $\mu(\cdot \mid x)$. For this section, assume

$$ x \sim \mathcal D, \qquad y \sim \mu(\cdot \mid x), $$

and that $A(x,y)$ is a deterministic function of the pair $(x,y)$.

Define the hard gate

$$ g_\theta(x,y) := \mathbf 1\!\left(c_\theta(x,y) \ge \tau(A(x,y))\right). $$

For each prompt $x$, let the acceptance probability be

$$ Z_\theta(x) := \sum_y \mu(y \mid x) g_\theta(x,y) = \mathbb E_{y \sim \mu(\cdot \mid x)}[g_\theta(x,y)]. $$

Assume $Z_\theta(x) > 0$ for $\mathcal D$-almost every $x$. Define the accepted-sample distribution

$$ \tilde \mu_\theta(y \mid x) := \frac{\mu(y \mid x) g_\theta(x,y)}{Z_\theta(x)}. $$

### Theorem 1

For every $x$ with $Z_\theta(x) > 0$, if $Y \sim \mu(\cdot \mid x)$ and we condition on the event $g_\theta(x,Y) = 1$, then the conditional law of $Y$ is exactly $\tilde \mu_\theta(\cdot \mid x)$:

$$ \mathbb P(Y = y \mid g_\theta(x,Y) = 1, x) = \tilde \mu_\theta(y \mid x). $$

**Proof.**

Fix $x$. By the definition of conditional probability,

$$ \mathbb P(Y = y \mid g_\theta(x,Y) = 1, x) = \frac{\mathbb P(Y = y, \, g_\theta(x,Y) = 1 \mid x)}{\mathbb P(g_\theta(x,Y) = 1 \mid x)}. $$

Since $Y \mid x \sim \mu(\cdot \mid x)$ and $g_\theta(x,y)$ is deterministic given $(x,y)$,

$$ \mathbb P(Y = y, \, g_\theta(x,Y) = 1 \mid x) = \mu(y \mid x) g_\theta(x,y), $$

and

$$ \mathbb P(g_\theta(x,Y) = 1 \mid x) = \sum_{y'} \mu(y' \mid x) g_\theta(x,y') = Z_\theta(x). $$

Substituting these two identities yields

$$ \mathbb P(Y = y \mid g_\theta(x,Y) = 1, x) = \frac{\mu(y \mid x) g_\theta(x,y)}{Z_\theta(x)} = \tilde \mu_\theta(y \mid x). $$

This proves the claim.

The theorem is elementary, but it matters. After gating, the effective training distribution is neither the original data distribution $\mu$ nor the current policy $\pi_\theta$. It is the conditional distribution obtained by restricting $\mu$ to the accepted set.

### Theorem 2

Fix any $x$ with $Z_\theta(x) > 0$. Then the exact divergence between the accepted distribution and the original data distribution is

$$ \mathrm{TV}(\tilde \mu_\theta(\cdot \mid x), \mu(\cdot \mid x)) = 1 - Z_\theta(x), $$

and

$$ D_{\mathrm{KL}}(\tilde \mu_\theta(\cdot \mid x) \| \mu(\cdot \mid x)) = \log \frac{1}{Z_\theta(x)}. $$

**Proof.**

Let $A_x := \{y : g_\theta(x,y) = 1\}$. Then

$$ \tilde \mu_\theta(y \mid x) = \begin{cases} \mu(y \mid x) / Z_\theta(x), & y \in A_x, \\ 0, & y \notin A_x. \end{cases} $$

For total variation distance,

$$ \mathrm{TV}(\tilde \mu_\theta, \mu) = \frac{1}{2}\sum_y \left|\tilde \mu_\theta(y \mid x) - \mu(y \mid x)\right|. $$

Split the sum over $A_x$ and its complement:

$$ \mathrm{TV}(\tilde \mu_\theta, \mu) = \frac{1}{2}\sum_{y \in A_x} \mu(y \mid x)\left(\frac{1}{Z_\theta(x)} - 1\right) + \frac{1}{2}\sum_{y \notin A_x} \mu(y \mid x). $$

Because $\sum_{y \in A_x} \mu(y \mid x) = Z_\theta(x)$ and $\sum_{y \notin A_x} \mu(y \mid x) = 1 - Z_\theta(x)$, this becomes

$$ \mathrm{TV}(\tilde \mu_\theta, \mu) = \frac{1}{2}(1 - Z_\theta(x)) + \frac{1}{2}(1 - Z_\theta(x)) = 1 - Z_\theta(x). $$

For the KL divergence,

$$ D_{\mathrm{KL}}(\tilde \mu_\theta \| \mu) = \sum_{y \in A_x} \tilde \mu_\theta(y \mid x) \log \frac{\tilde \mu_\theta(y \mid x)}{\mu(y \mid x)} = \sum_{y \in A_x} \frac{\mu(y \mid x)}{Z_\theta(x)} \log \frac{1}{Z_\theta(x)} = \log \frac{1}{Z_\theta(x)}. $$

This proves both identities.

An immediate corollary is the following generic bound.

### Corollary 3

For every $x$ with $Z_\theta(x) > 0$,

$$ \mathrm{TV}(\tilde \mu_\theta(\cdot \mid x), \pi_\theta(\cdot \mid x)) \le 1 - Z_\theta(x) + \mathrm{TV}(\mu(\cdot \mid x), \pi_\theta(\cdot \mid x)). $$

**Proof.**

Apply the triangle inequality:

$$ \mathrm{TV}(\tilde \mu_\theta, \pi_\theta) \le \mathrm{TV}(\tilde \mu_\theta, \mu) + \mathrm{TV}(\mu, \pi_\theta), $$

and substitute Theorem 2.

This bound is intentionally modest. It does **not** say that gating always makes the accepted distribution closer to $\pi_\theta$ than $\mu$ was. That stronger statement is false without additional assumptions on how well the gate is calibrated.

## 6. Relation to the On-Policy Surrogate

Now consider the training utility

$$ u_\theta(x,y) := A(x,y)\sum_{i=1}^T \ell_{\theta,i}(x,y). $$

This is the quantity that appears in the standard log-probability surrogate. Define three different objectives:

$$ J_{\mathrm{on}}(\theta) := \mathbb E_{x \sim \mathcal D, \, y \sim \pi_\theta(\cdot \mid x)}[u_\theta(x,y)], $$

$$ \tilde J(\theta) := \mathbb E_{x \sim \mathcal D, \, y \sim \tilde \mu_\theta(\cdot \mid x)}[u_\theta(x,y)], $$

and

$$ J_{\mathrm{gate}}(\theta) := \mathbb E_{x \sim \mathcal D, \, y \sim \mu(\cdot \mid x)}[g_\theta(x,y) u_\theta(x,y)]. $$

These quantities are not the same:

1. $J_{\mathrm{on}}$ is the on-policy surrogate one would ideally optimize.
2. $\tilde J$ is the surrogate under the accepted-sample distribution.
3. $J_{\mathrm{gate}}$ is the raw objective computed directly from data samples and the gate.

### Theorem 4

Let

$$ \tilde J_x(\theta) := \mathbb E_{y \sim \tilde \mu_\theta(\cdot \mid x)}[u_\theta(x,y)]. $$

Then

$$ J_{\mathrm{gate}}(\theta) = \mathbb E_{x \sim \mathcal D}[Z_\theta(x)\tilde J_x(\theta)]. $$

**Proof.**

For each fixed $x$,

$$ \tilde J_x(\theta) = \sum_y \tilde \mu_\theta(y \mid x) u_\theta(x,y) = \sum_y \frac{\mu(y \mid x) g_\theta(x,y)}{Z_\theta(x)} u_\theta(x,y). $$

Multiplying both sides by $Z_\theta(x)$ gives

$$ Z_\theta(x)\tilde J_x(\theta) = \sum_y \mu(y \mid x) g_\theta(x,y) u_\theta(x,y) = \mathbb E_{y \sim \mu(\cdot \mid x)}[g_\theta(x,y)u_\theta(x,y)]. $$

Taking expectation over $x \sim \mathcal D$ yields the result.

This identity makes the role of the acceptance rate explicit: the raw gate objective is the accepted-distribution objective weighted by the probability of passing the gate.

### Proposition 5

Assume there exists a constant $B < \infty$ such that

$$ |u_\theta(x,y)| \le B $$

for all relevant $(x,y)$. Then

$$ |\tilde J(\theta) - J_{\mathrm{on}}(\theta)| \le 2B \, \mathbb E_{x \sim \mathcal D}\left[\mathrm{TV}(\tilde \mu_\theta(\cdot \mid x), \pi_\theta(\cdot \mid x))\right]. $$

Consequently, by Corollary 3,

$$ |\tilde J(\theta) - J_{\mathrm{on}}(\theta)| \le 2B \, \mathbb E_{x \sim \mathcal D}\left[1 - Z_\theta(x) + \mathrm{TV}(\mu(\cdot \mid x), \pi_\theta(\cdot \mid x))\right]. $$

**Proof.**

For each fixed $x$, let $P_x := \tilde \mu_\theta(\cdot \mid x)$ and $Q_x := \pi_\theta(\cdot \mid x)$. Then

$$ \left|\mathbb E_{P_x}[u_\theta(x,\cdot)] - \mathbb E_{Q_x}[u_\theta(x,\cdot)]\right| \le \sum_y |u_\theta(x,y)| \, |P_x(y) - Q_x(y)| \le B \sum_y |P_x(y) - Q_x(y)|. $$

Using $\sum_y |P_x(y) - Q_x(y)| = 2 \, \mathrm{TV}(P_x,Q_x)$ gives

$$ \left|\mathbb E_{P_x}[u_\theta(x,\cdot)] - \mathbb E_{Q_x}[u_\theta(x,\cdot)]\right| \le 2B \, \mathrm{TV}(P_x,Q_x). $$

Taking expectation over $x \sim \mathcal D$ proves the first inequality. The second follows from Corollary 3.

The proposition is the right level of rigor for CGPO. It does **not** claim that maximizing the gate objective necessarily improves the true RL objective. It says that the surrogate mismatch is controlled when two quantities are small:

1. the gate discards little probability mass, so $Z_\theta(x)$ stays close to $1$;
2. the data distribution $\mu(\cdot \mid x)$ is already close to the current policy.

That is a bias-control statement, not a monotonic-improvement theorem.

## 7. A Crucial Optimization Detail

The literal hard-gated objective is not smoothly differentiable with respect to $\theta$, because the gate itself depends on $\theta$ through an indicator function. A practical implementation therefore has to decide how gradients flow.

There are two principled choices.

### 7.1 Detached hard masks

Compute the gate in the forward pass, but stop gradients through the gate:

$$ \bar g_\theta(x,y,A) := \operatorname{sg}\!\left[g_\theta(x,y,A)\right], \qquad \bar g_{\theta,i}(x,y,A) := \operatorname{sg}\!\left[g_{\theta,i}(x,y,A)\right], $$

where $\operatorname{sg}[\cdot]$ denotes stop-gradient. Then optimize

$$ \mathcal L_{\mathrm{hard}}(\theta) := \mathbb E_{(x,y,A) \sim \mathcal B}\left[A(x,y) \, \bar g_\theta(x,y,A) \sum_{i=1}^T \bar g_{\theta,i}(x,y,A) \, \ell_{\theta,i}(x,y)\right]. $$

This corresponds to a training rule that recomputes the active sample set at each forward pass, but does not backpropagate through the discrete selection event itself.

### 7.2 Soft gates

Replace the indicator with a smooth approximation. If we define $\kappa(A) := \log \tau(A)$ and $\kappa_i(A) := \log \tau_i(A)$, one option is

$$ s_\theta(x,y,A) := \sigma\!\left(\alpha(\bar \ell_\theta(x,y) - \kappa(A))\right), \qquad s_{\theta,i}(x,y,A) := \sigma\!\left(\alpha_i(\ell_{\theta,i}(x,y) - \kappa_i(A))\right), $$

with sharpness parameters $\alpha,\alpha_i > 0$. The soft version becomes

$$ \mathcal L_{\mathrm{soft}}(\theta) := \mathbb E_{(x,y,A) \sim \mathcal B}\left[A(x,y) \, s_\theta(x,y,A) \sum_{i=1}^T s_{\theta,i}(x,y,A) \, \ell_{\theta,i}(x,y)\right]. $$

Soft gates trade away exact sample selection in exchange for smooth optimization. Hard gates preserve the intended semantics more directly, but they should be implemented with detached masks if one wants stable first-order training.

## 8. Why the Asymmetric Threshold Is Reasonable

The positive/negative asymmetry is not cosmetic. It reflects a real asymmetry in LLM outputs.

If $A(x,y) > 0$, the response is useful. There may be many semantically correct responses with different wording, structure, or level of detail. A permissive threshold $\tau_+$ allows CGPO to preserve that diversity.

If $A(x,y) < 0$, the response is harmful. But not every harmful response deserves equal weight. A low-probability error that the current model almost never produces is not the most urgent target for optimization. A stricter threshold $\tau_-$ focuses training on errors that are both bad and still plausible under the current policy.

In that sense, CGPO acts like a hard-negative mining rule defined directly in policy space.

## 9. A Practical Default Recipe

For a first implementation, the following version is a sensible default.

1. Use average token log-probability $\bar \ell_\theta(x,y)$, or equivalently the geometric-mean score $c_\theta(x,y)$, as the sequence-level compatibility metric.
2. Use both a sequence gate and token gates.
3. Choose thresholds such that $\tau_+ < \tau_-$ and $\tau_{+,i} < \tau_{-,i}$.
4. Feed in a normalized scalar signal $A(x,y)$ rather than raw uncentered rewards.
5. Monitor the empirical acceptance rate $\hat Z := \frac{1}{B}\sum_{j=1}^B g_\theta(x^{(j)},y^{(j)},A^{(j)})$ during training.

The corresponding hard-mask objective is

$$ \mathcal L_{\mathrm{CGPO}}(\theta) := \mathbb E_{(x,y,A) \sim \mathcal B}\left[A(x,y) \, \bar g_\theta(x,y,A) \sum_{i=1}^{|y|} \bar g_{\theta,i}(x,y,A) \, \log \pi_\theta(y_i \mid x, y_{ <i })\right]. $$

Monitoring $\hat Z$ is especially important. If it collapses toward zero, then either the thresholds are too strict or the data distribution is too stale relative to the current policy. In either case, the bounds above predict a larger surrogate mismatch.

## 10. Limitations

CGPO is useful, but it should be described honestly.

1. It is a biased surrogate, not an unbiased correction of off-policy sampling.
2. Its behavior depends strongly on threshold design.
3. The active training set is coupled to the current policy, which can make optimization dynamics discontinuous under hard masks.
4. If the data distribution is far from the current policy, acceptance can collapse and training signal can disappear.
5. The theory above controls distribution shift and surrogate mismatch; it does not prove superior final task performance.

These are not minor caveats. They are part of the method definition.

## 11. When CGPO Is a Good Fit

CGPO is most compelling when the bottleneck is not lack of a ratio-clipping method, but the operational burden of maintaining a precise behavior-policy reference for every sample. That is common in:

1. mixed online and replay-buffer training;
2. heterogeneous data pools assembled from multiple checkpoints or filters;
3. pipelines where the most important negatives are the mistakes the current model still tends to make.

In those settings, "is this sample compatible with the current policy?" can be a more natural question than "what exact ratio should be formed against the policy that generated it?"

## 12. Conclusion

Compatibility-Gated Policy Optimization replaces ratio clipping with direct compatibility testing under the current policy. The method is easy to state:

1. score each sample using current-policy token probabilities;
2. keep only samples whose compatibility exceeds a threshold;
3. use stricter gates for negative samples than for positive ones.

What is less obvious, and therefore worth stating precisely, is the mathematical status of the method. A hard gate induces an accepted-sample distribution that can be written exactly, its divergence from the source distribution is controlled exactly by the acceptance rate, and its mismatch with the on-policy surrogate can be bounded under standard assumptions. Those are rigorous statements. Stronger claims, such as guaranteed improvement of the original RL objective, require additional assumptions and should be proved separately rather than implied by the gating mechanism alone.

Viewed this way, CGPO is best understood as a deliberately biased, engineering-friendly surrogate objective for LLM RL: simpler than ratio tracking, naturally asymmetric between positive and negative samples, and most useful when current-policy compatibility is the right notion of sample relevance.
