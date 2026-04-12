---
date: '2026-04-11T10:31:01+08:00'
draft: false
title: 'An Exact Bias Correction for Truncated Bernoulli Groups'
math: true
---

## Background

Recent work such as [Your Group-Relative Advantage Is Biased](https://arxiv.org/abs/2601.08521) points out a simple but important issue in group-based RL for LLMs. For one prompt, we sample a group of responses, score them, and then construct a relative training signal from that group. In practice, groups that are entirely correct or entirely incorrect are often dropped, because they contain no within-group ranking information.

That filtering step changes the statistical problem. Once we condition on keeping only the non-degenerate groups, the ordinary sample mean is no longer an unbiased estimator of the latent correctness probability.

This post studies the cleanest version of that question:

> If we sample $n$ i.i.d. Bernoulli variables and discard the all-zero and all-one groups, can we still estimate the original Bernoulli mean exactly?

For this problem, the answer is complete:

1. If the group size $n$ is odd, there is a closed-form unbiased estimator.
2. If the group size $n$ is even, no unbiased estimator exists.

## Problem Setup

Let
$$
A_1, \dots, A_n \overset{\mathrm{i.i.d.}}{\sim} \mathrm{Bernoulli}(p),
$$
and define the count
$$
K := \sum_{i=1}^n A_i.
$$

We keep only non-degenerate groups:
$$
S := \{1 \le K \le n-1\}.
$$
Equivalently, we discard the cases $K=0$ and $K=n$.

The survival probability is
$$
\mathbb{P}_p(S) = 1 - (1-p)^n - p^n.
$$
Conditioned on $S$, the count $K$ has the truncated binomial distribution
$$
\mathbb{P}_p(K=k \mid S) = \frac{\binom{n}{k}p^k(1-p)^{n-k}}{1-(1-p)^n-p^n}, \qquad k=1,\dots,n-1.
$$

Our goal is to find a statistic $T=T(A_1,\dots,A_n)$ such that
$$
\mathbb{E}_p[T \mid S] = p \qquad \text{for all } p \in (0,1).
$$

## The Naive Mean Becomes Biased

Without truncation, the sample mean
$$
\bar A := \frac{K}{n}
$$
is unbiased for $p$. Under the conditional law given $S$, however,
$$
\mathbb{E}_p[\bar A \mid S] = \frac{\mathbb{E}_p[\bar A \mathbf{1}_S]}{\mathbb{P}_p(S)} = \frac{p - p^n}{1-(1-p)^n-p^n}.
$$
This is generally not equal to $p$.

So the bias is not a numerical artifact. It is a structural consequence of conditioning on the event that a group contains at least one success and at least one failure.

## Reduction to a Function of the Count

Even though an estimator could depend on the whole vector $(A_1,\dots,A_n)$, it is enough to study estimators of the form $h(K)$.

Indeed, conditioned on $K=k$, every binary vector with exactly $k$ ones is equally likely, and this conditional distribution does not depend on $p$. Therefore, if $T$ is any estimator, then its Rao-Blackwellization
$$
h(K) := \mathbb{E}[T \mid K]
$$
satisfies
$$
\mathbb{E}_p[h(K) \mid S] = \mathbb{E}_p[T \mid S].
$$
So the existence question reduces to finding a function $h:\{1,\dots,n-1\} \to \mathbb{R}$ such that
$$
\sum_{k=1}^{n-1} h(k)\binom{n}{k}p^k(1-p)^{n-k} = p\bigl[1-(1-p)^n-p^n\bigr] \qquad \forall p \in (0,1). \tag{1}
$$

This is a polynomial identity in $p$.

## Main Theorem

Let
$$
B_{k,n}(p) := \binom{n}{k}p^k(1-p)^{n-k}
$$
be the Bernstein basis polynomial of degree $n$.

### Theorem

For the truncated Bernoulli model above:

1. If $n$ is even, there is no estimator $T$ such that $\mathbb{E}_p[T \mid S]=p$ for every $p \in (0,1)$.
2. If $n$ is odd, there is a unique unbiased estimator among all functions of $K$, namely
$$
\widehat p_n := \frac{K}{n} - \frac{(-1)^{K-1}}{\binom{n}{K}}, \qquad K \in \{1,\dots,n-1\}.
$$

## Why Even Group Sizes Are Impossible

The left-hand side of (1) is a linear combination of $B_{k,n}(p)$ for $k=1,\dots,n-1$, so it is a polynomial of degree at most $n$.

Now expand the right-hand side:
$$
p\bigl[1-(1-p)^n-p^n\bigr] = p - p(1-p)^n - p^{n+1}.
$$
The term $p(1-p)^n$ contributes $(-1)^n p^{n+1}$ at the highest degree, so the coefficient of $p^{n+1}$ on the right-hand side is
$$
-1 - (-1)^n.
$$
Therefore:

1. If $n$ is even, that coefficient is $-2$, so the right-hand side has degree $n+1$.
2. The left-hand side still has degree at most $n$.

Hence equality in (1) is impossible when $n$ is even. No unbiased estimator exists.

## Closed-Form Solution for Odd Group Sizes

Assume now that $n$ is odd. Then
$$
-1 - (-1)^n = 0,
$$
so the right-hand side of (1) has degree at most $n$.

Because the Bernstein polynomials $\{B_{k,n}\}_{k=0}^n$ form a basis of all polynomials of degree at most $n$, the coefficient vector in that basis is unique. Also,
$$
p\bigl[1-(1-p)^n-p^n\bigr] = 0 \qquad \text{at } p=0 \text{ and } p=1,
$$
so the coefficients of $B_{0,n}$ and $B_{n,n}$ must be zero. This determines a unique function $h(k)$ for $k=1,\dots,n-1$.

To get it in closed form, use two identities.

First,
$$
\sum_{k=1}^{n-1} \frac{k}{n} B_{k,n}(p) = \sum_{k=0}^{n} \frac{k}{n} B_{k,n}(p) - B_{n,n}(p) = p - p^n.
$$

Second,
$$
\sum_{k=1}^{n-1} \frac{(-1)^{k-1}}{\binom{n}{k}} B_{k,n}(p) = \sum_{k=1}^{n-1} (-1)^{k-1} p^k(1-p)^{n-k}.
$$
When $n$ is odd, $n-1$ is even, and this finite geometric sum becomes
$$
\begin{aligned}
\sum_{k=1}^{n-1} (-1)^{k-1} p^k(1-p)^{n-k}
&= p(1-p)^{n-1}\sum_{j=0}^{n-2}\left(-\frac{p}{1-p}\right)^j \\
&= p(1-p)^n - p^n + p^{n+1}.
\end{aligned}
$$

Subtracting the two identities gives
$$
\begin{aligned}
\sum_{k=1}^{n-1} \left(\frac{k}{n} - \frac{(-1)^{k-1}}{\binom{n}{k}}\right) B_{k,n}(p)
& = (p-p^n) - \bigl[p(1-p)^n - p^n + p^{n+1}\bigr] \\
& = p\bigl[1-(1-p)^n-p^n\bigr].
\end{aligned}
$$
So the function
$$
h(k) = \frac{k}{n} - \frac{(-1)^{k-1}}{\binom{n}{k}}
$$
satisfies (1), and therefore
$$
\mathbb{E}_p[\widehat p_n \mid S] = p.
$$

## Examples

For small odd group sizes, the estimator is especially simple.

For $n=3$,
$$
\widehat p_3 =
\begin{cases}
0, & K=1, \\
1, & K=2.
\end{cases}
$$

For $n=5$,
$$
\widehat p_5 =
\begin{cases}
0, & K=1, \\
\tfrac{1}{2}, & K=2, \\
\tfrac{1}{2}, & K=3, \\
1, & K=4.
\end{cases}
$$

For $n=7$,
$$
\widehat p_7 =
\begin{cases}
0, & K=1, \\
\tfrac{1}{3}, & K=2, \\
\tfrac{2}{5}, & K=3, \\
\tfrac{3}{5}, & K=4, \\
\tfrac{2}{3}, & K=5, \\
1, & K=6.
\end{cases}
$$

## What This Means for Advantage Estimation

At the level of this Bernoulli abstraction, the picture is exact:

1. If the group size is odd, the truncation bias can be removed analytically by replacing the retained-group mean with $\widehat p_n$.
2. If the group size is even, no estimator can be unbiased for all $p$ after this truncation. Any correction must therefore be approximate, prior-dependent, or based on changing the sampling procedure.

In particular, if one insists on exact unbiasedness, then an even group size is fundamentally incompatible with dropping all-zero and all-one groups.

It is also worth being precise about scope. The result here solves the scalar estimation problem for the latent Bernoulli mean. It does not automatically imply that every downstream RL "advantage" constructed from that estimate is unbiased in a policy-gradient sense. But it completely characterizes what is possible at the level of truncated Bernoulli statistics.

## Takeaway

Discarding degenerate groups is not an innocent preprocessing step. It changes the sampling law, and the ordinary group mean becomes biased under the retained distribution.

The full answer is:

1. Odd group size: an exact closed-form correction exists,
$$
\widehat p_n = \frac{K}{n} - \frac{(-1)^{K-1}}{\binom{n}{K}}.
$$
2. Even group size: exact unbiased recovery is impossible.

So if the training pipeline drops all-correct and all-incorrect groups and you still want a mathematically exact correction, the parity of the group size is the deciding factor.

## Beyond Bernoulli: Multi-Level Scores

The Bernoulli model is the two-level case: each score is either $0$ or $1$. A natural extension is to allow a finite score set
$$
0 = v_0 < v_1 < \cdots < v_m = 1,
$$
where $m \ge 2$, so there is at least one interior score level.

Let
$$
X_1,\dots,X_n \overset{\mathrm{i.i.d.}}{\sim} \mathbb{P}(X=v_j)=\pi_j, \qquad j=0,\dots,m,
$$
and denote the population mean by
$$
\mu := \mathbb{E}[X] = \sum_{j=0}^m v_j \pi_j.
$$

The most natural analogue of the Bernoulli truncation is to discard groups with no within-group variation, namely
$$
S_{\mathrm{multi}} := \{\text{the sample } X_1,\dots,X_n \text{ is not constant}\}.
$$
Equivalently, if $N_j := \sum_{i=1}^n \mathbf{1}\{X_i=v_j\}$ is the count of level $v_j$, then we keep only outcomes for which no $N_j$ equals $n$.

The question is the same as before: can we find an estimator $T$ such that
$$
\mathbb{E}[T \mid S_{\mathrm{multi}}] = \mu
$$
for all probability vectors $(\pi_0,\dots,\pi_m)$?

### A Strong Negative Result

For this multi-level model, the Bernoulli miracle disappears.

### Theorem

Assume the score support contains at least three distinct levels, i.e. $m \ge 2$. Then for every group size $n \ge 2$, there is no estimator $T$ satisfying
$$
\mathbb{E}[T \mid S_{\mathrm{multi}}] = \mu
$$
for all choices of $(\pi_0,\dots,\pi_m)$.

### Proof

We prove the theorem by contradiction. Assume that there exists an estimator $T$, defined on the retained event $S_{\mathrm{multi}}$, such that
$$
\mathbb{E}_\pi[T \mid S_{\mathrm{multi}}] = \mu \qquad \text{for every probability vector } \pi=(\pi_0,\dots,\pi_m),
$$
where
$$
\mu = \sum_{j=0}^m v_j \pi_j.
$$

It is enough to work on the interior of the simplex
$$
\Delta_m^\circ := \left\{\pi \in (0,1)^{m+1} : \sum_{j=0}^m \pi_j = 1\right\},
$$
because an estimator that is unbiased for all probability vectors is, in particular, unbiased for every $\pi \in \Delta_m^\circ$.

Let
$$
N=(N_0,\dots,N_m), \qquad N_j := \sum_{i=1}^n \mathbf{1}\{X_i=v_j\},
$$
and define
$$
\mathcal{C} := \left\{ c=(c_0,\dots,c_m)\in \mathbb{N}_0^{m+1} : \sum_{j=0}^m c_j = n,\ \max_j c_j < n \right\}.
$$
Thus $S_{\mathrm{multi}}=\{N \in \mathcal{C}\}$.

For each $c \in \mathcal{C}$, conditional on the event $N=c$, every sequence $(X_1,\dots,X_n)$ with exactly $c_j$ occurrences of $v_j$ is equally likely. This conditional law depends only on $c$ and not on $\pi$. Therefore the Rao-Blackwellized statistic
$$
h(c) := \mathbb{E}_\pi[T \mid N=c], \qquad c \in \mathcal{C},
$$
is well defined independently of the choice of $\pi \in \Delta_m^\circ$.

Since $N=c$ implies $S_{\mathrm{multi}}$, the tower property gives
$$
\mathbb{E}_\pi[T\mathbf{1}_{S_{\mathrm{multi}}}] = \mathbb{E}_\pi[h(N)\mathbf{1}_{S_{\mathrm{multi}}}] = \sum_{c\in\mathcal{C}} h(c)\,\mathbb{P}_\pi(N=c).
$$
Using the multinomial law,
$$
\mathbb{P}_\pi(N=c) = \binom{n}{c_0,\dots,c_m}\prod_{j=0}^m \pi_j^{c_j}, \qquad c\in\mathcal{C}.
$$
On the other hand,
$$
\mathbb{E}_\pi[T\mathbf{1}_{S_{\mathrm{multi}}}] = \mathbb{E}_\pi[T \mid S_{\mathrm{multi}}]\mathbb{P}_\pi(S_{\mathrm{multi}}) = \mu\left(1-\sum_{j=0}^m \pi_j^n\right),
$$
because $\mathbb{P}_\pi(S_{\mathrm{multi}})=1-\sum_{j=0}^m \pi_j^n$.
Hence, for every $\pi \in \Delta_m^\circ$,
$$
\sum_{c\in\mathcal{C}} h(c)\binom{n}{c_0,\dots,c_m}\prod_{j=0}^m \pi_j^{c_j} = \mu\left(1-\sum_{j=0}^m \pi_j^n\right). \tag{2}
$$

Now write
$$
q_r := \pi_r \quad (r=1,\dots,m), \qquad s(q) := \sum_{r=1}^m q_r, \qquad \pi_0 = 1-s(q).
$$
Then (2) becomes an identity on the open set
$$
U := \{q \in \mathbb{R}^m : q_r>0,\ s(q)<1\}.
$$
Define
$$
L(q) := \sum_{c\in\mathcal{C}} h(c)\binom{n}{c_0,\dots,c_m} (1-s(q))^{c_0}\prod_{j=1}^m q_j^{c_j},
$$
and
$$
R(q) := \left(\sum_{j=1}^m v_j q_j\right) \left[ 1-(1-s(q))^n-\sum_{j=1}^m q_j^n \right].
$$
Then $L(q)=R(q)$ for all $q\in U$.

Both $L$ and $R$ are polynomials in $(q_1,\dots,q_m)$. Since two polynomials that agree on a nonempty open set must agree identically, we may regard $L\equiv R$ as a polynomial identity on all of $\mathbb{R}^m$.

Next we compare degrees.

For each $c\in\mathcal{C}$, the factor
$$
(1-s(q))^{c_0}\prod_{j=1}^m q_j^{c_j}
$$
expands into monomials of total degree at most
$$
c_0 + c_1 + \cdots + c_m = n.
$$
Therefore
$$
\deg L \le n.
$$

For the right-hand side, note that
$$
1-(1-s)^n = ns - \binom{n}{2}s^2 + \cdots + (-1)^{n+1}s^n.
$$
Hence the homogeneous degree-$n$ part of the bracket in $R(q)$ is
$$
(-1)^{n+1}s(q)^n - \sum_{j=1}^m q_j^n.
$$
Multiplying by the linear factor $\sum_{j=1}^m v_j q_j$, the homogeneous degree-$(n+1)$ part of $R$ is
$$
H(q) := \left(\sum_{j=1}^m v_j q_j\right) \left[(-1)^{n+1}s(q)^n - \sum_{j=1}^m q_j^n\right].
$$

We claim that $H$ is not the zero polynomial when $m\ge 2$ and $n\ge 2$. Indeed, evaluate it on the line
$$
q_1=q_2=t,\qquad q_3=\cdots=q_m=0,
$$
with $0<t<1/2$. Then $s(q)=2t$, so
$$
\begin{aligned}
H(q)
&= (v_1+v_2)t\left[(-1)^{n+1}(2t)^n - t^n - t^n\right] \\
&= (v_1+v_2)\bigl[(-1)^{n+1}2^n - 2\bigr] t^{n+1}.
\end{aligned}
$$
Since $v_1,v_2>0$ and $n\ge 2$, the coefficient
$$
(-1)^{n+1}2^n - 2
$$
is nonzero:
$$
\begin{cases}
-2^n-2 \neq 0, & n \text{ even},\\[4pt]
2^n-2 \neq 0, & n \text{ odd}.
\end{cases}
$$
Thus $H \not\equiv 0$. Therefore $R$ has total degree $n+1$:
$$
\deg R = n+1.
$$

This contradicts $\deg L \le n$ and the identity $L\equiv R$. The contradiction shows that no such estimator $T$ can exist.

Therefore, when the score support has at least three distinct levels, there is no estimator satisfying
$$
\mathbb{E}[T \mid S_{\mathrm{multi}}] = \mu
$$
for all choices of $(\pi_0,\dots,\pi_m)$.

### Interpretation

This shows that the Bernoulli case is genuinely exceptional. The exact cancellation that makes odd group sizes solvable relies on the fact that the support has only two points. Once the score support has three or more levels, the same truncation problem becomes impossible in a much stronger sense: there is no exact unbiased correction at all, regardless of whether $n$ is odd or even.

So under the natural generalization from binary scores to finitely many score buckets, the final answer is:

1. Two score levels: the Bernoulli result above applies, and odd group sizes admit an exact correction.
2. Three or more score levels: no analogous uniformly unbiased estimator exists after discarding all constant-score groups.

If one uses a different truncation rule, the algebra changes. For example, if one discards only all-$0$ and all-$1$ groups but keeps groups that are constant at an interior score level, then the impossibility proof above no longer applies directly. But for the most natural extension of the Bernoulli setup, the conclusion is negative.


## More Rollouts Reduce the Bias Envelope

The impossibility results above concern exact unbiasedness. They do not imply that the ordinary retained-group mean stays badly biased when the rollout count $n$ grows. The correct statement is more precise.

In this section, "more accurate" means "having smaller absolute bias relative to the original population mean." No claim about variance or mean-squared error is being made here.

Let
$$
X_1,\dots,X_n
$$
be i.i.d. with finite support
$$
0 \le v_0 < v_1 < \cdots < v_m \le 1,
$$
and probabilities
$$
\mathbb{P}(X=v_j)=\pi_j, \qquad j=0,\dots,m,
$$
where at least two of the $\pi_j$ are positive. Define
$$
\mu := \mathbb{E}[X] = \sum_{j=0}^m v_j \pi_j,
\qquad
\bar X_n := \frac{1}{n}\sum_{i=1}^n X_i,
$$
and let
$$
S_n := \{X_1,\dots,X_n \text{ are not all equal}\}.
$$
In the Bernoulli case, $S_n$ is exactly the event $S=\{1 \le K \le n-1\}$ studied above.

### Proposition

For every $n \ge 2$,
$$
\mathbb{E}[\bar X_n \mid S_n] = \frac{\mu - \sum_{j=0}^m v_j \pi_j^n}{1-\sum_{j=0}^m \pi_j^n}.
$$
Therefore the bias of the retained-group mean is
$$
b_n := \mathbb{E}[\bar X_n \mid S_n] - \mu = \frac{\sum_{j=0}^m (\mu-v_j)\pi_j^n}{1-\sum_{j=0}^m \pi_j^n}. \tag{3}
$$

If we write
$$
\delta_n := \sum_{j=0}^m \pi_j^n, \qquad \alpha := \max_{0 \le j \le m} \pi_j,
$$
then $\alpha<1$ and
$$
|b_n| \le \frac{\delta_n}{1-\delta_n} \le \frac{\alpha^{n-1}}{1-\alpha^{n-1}}. \tag{4}
$$
Since $\alpha \in (0,1)$, the right-hand side of (4) is strictly decreasing in $n$ and converges to $0$ exponentially fast. Hence, for every fixed non-degenerate law, the truncation bias of the retained-group mean is bounded by a strictly decreasing function of the rollout count.

### Proof

Because $S_n$ is permutation-invariant and the sample is i.i.d., exchangeability gives
$$
\mathbb{E}[\bar X_n \mid S_n] = \mathbb{E}[X_1 \mid S_n].
$$
Also,
$$
\mathbb{E}[X_1 \mathbf{1}_{S_n}] = \mathbb{E}[X_1] - \mathbb{E}[X_1 \mathbf{1}_{\{X_1=\cdots=X_n\}}].
$$
Now
$$
\mathbb{E}[X_1] = \mu,
$$
and
$$
\mathbb{E}[X_1 \mathbf{1}_{\{X_1=\cdots=X_n\}}] = \sum_{j=0}^m v_j \,\mathbb{P}(X_1=\cdots=X_n=v_j) = \sum_{j=0}^m v_j \pi_j^n.
$$
Therefore
$$
\mathbb{E}[X_1 \mathbf{1}_{S_n}] = \mu - \sum_{j=0}^m v_j \pi_j^n.
$$
Similarly,
$$
\mathbb{P}(S_n) = 1-\sum_{j=0}^m \mathbb{P}(X_1=\cdots=X_n=v_j) = 1-\sum_{j=0}^m \pi_j^n.
$$
Dividing the last two identities yields
$$
\mathbb{E}[\bar X_n \mid S_n] = \frac{\mu - \sum_{j=0}^m v_j \pi_j^n}{1-\sum_{j=0}^m \pi_j^n},
$$
which proves (3).

Next, since $0 \le \mu \le 1$ and $0 \le v_j \le 1$, we have
$$
|\mu-v_j| \le 1 \qquad \text{for all } j.
$$
Hence
$$
\left|\sum_{j=0}^m (\mu-v_j)\pi_j^n\right| \le \sum_{j=0}^m |\mu-v_j| \pi_j^n \le \sum_{j=0}^m \pi_j^n = \delta_n.
$$
Using (3), we obtain
$$
|b_n| \le \frac{\delta_n}{1-\delta_n}.
$$
Finally,
$$
\delta_n = \sum_{j=0}^m \pi_j \pi_j^{\,n-1} \le \alpha^{n-1}\sum_{j=0}^m \pi_j = \alpha^{n-1}.
$$
Substituting this into the previous inequality proves (4). Since at least two of the $\pi_j$ are positive, we have $\alpha<1$, and thus $\alpha^{n-1}/(1-\alpha^{n-1})$ is strictly decreasing in $n$ and converges to $0$. This completes the proof.

### Bernoulli Specialization

For the $0$-$1$ model,
$$
v_0=0,\qquad v_1=1,\qquad \pi_1=p,\qquad \pi_0=1-p,
$$
so (3) becomes
$$
b_n(p) = \mathbb{E}[\bar A \mid S]-p = \frac{p(1-p)\bigl[(1-p)^{n-1}-p^{n-1}\bigr]}{1-(1-p)^n-p^n}.
$$
Also, with
$$
\alpha = \max\{p,1-p\}<1,
$$
the general bound (4) gives
$$
|b_n(p)| \le \frac{\alpha^{n-1}}{1-\alpha^{n-1}}.
$$
Thus in the Bernoulli model, exact unbiased correction is delicate, but the bias of the naive retained-group mean is still exponentially suppressed as the rollout count grows.

### Multi-Level Specialization

For the finite multi-level model studied in the previous section, exact unbiased estimation is impossible for every $n \ge 2$. Nevertheless, the same bias formula (3) and the same exponential bound (4) remain valid. Therefore, although exact unbiased recovery fails, the naive retained-group mean still becomes asymptotically unbiased:
$$
\lim_{n\to\infty} \left|\mathbb{E}[\bar X_n \mid S_n]-\mu\right| = 0.
$$

### Important Precision

For general multi-level rewards, the stronger statement
$$
|b_{n+1}| \le |b_n| \qquad \text{for every } n
$$
is false in general. The universally correct statement is the decreasing upper bound (4).

Indeed, take support
$$
(v_0,v_1,v_2,v_3)=\left(0,\frac14,\frac12,1\right)
$$
and probabilities
$$
(\pi_0,\pi_1,\pi_2,\pi_3)=\left(\frac1{10},\frac1{10},\frac12,\frac3{10}\right).
$$
Then
$$
\mu = \frac{23}{40},
$$
and a direct substitution into (3) gives
$$
|b_3| = \frac{1}{705} < \frac{267}{185840} = |b_4|.
$$
So pointwise monotonicity of the absolute bias is not valid for every multi-level law. What holds without exception is the exponentially decaying envelope (4), and that is the mathematically correct sense in which more rollouts reduce the truncation error.
