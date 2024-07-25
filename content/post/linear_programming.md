---
title: "How to Solve Linear Programming Efficiently?"
date: 2024-07-22T16:58:56+08:00
draft: false
math: true
---

- [Background](#background)
- [Simplex Method](#simplex-method)
- [Interior-Point Method](#interior-point-method)
- [ALM and ADMM](#alm-and-admm)
- [PDHG](#pdhg)

## Background

One fundamental question in the field of optimization is how to solve linear programming (LP) problems. Typically, there are two well-known frameworks, namely the Simplex Method and the Interior-Point Method, that have been proposed to tackle LP problems.

However, when it comes to solving large-scale problems with millions (even billions) of variables and constraints, both the Simplex Method and the Interior-Point Method may not be practical enough. Their efficiency tends to decrease significantly under such circumstances. If you're interested, you can evaluate the performance of the simplex method and interior point method using the Python package `scipy.optimize`.

Given these challenges, it becomes crucial to explore alternative approaches that can provide approximate solutions to LP problems within an acceptable time frame. In the following discussion, we will delve into this topic and explore potential methods for finding approximate solutions efficiently.

For simplicity, in this article, we only consider the standard formulation LP problem, i.e., 
$$
\begin{aligned}
& \min_x & \quad \langle c, x \rangle \\\
& \textrm{s.t.} & \quad Ax = b, \\\
& & \quad x \geq 0.
\end{aligned}
$$
Here, we assume that $A \in \mathbb{R}^{m \times n}$ and $A$ is full-rank. 

It is assumed that the linear programming (LP) problem is feasible, meaning there exists at least one solution to it. If you're interested in further studying the topic, you can refer to Stephen J. Wright's book, [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5), or Yinyu Ye's book, [Linear and Nonlinear Programming](https://link.springer.com/book/10.1007/978-0-387-74503-9). These books provide comprehensive information and insights into the respective subjects.

## Simplex Method 

The simplex method is indeed known to have exponential worst-case complexity, but it is highly efficient for small-scale linear programming (LP) problems. For instances where the number of constraints (m) and variables (n) is not larger than around $10^3$, the simplex method can be the fastest algorithm available.

However, when dealing with larger LP problems, even with m and n on the order of $10^4$, the simplex method tends to be slower compared to interior point methods. Interior point methods have demonstrated better performance for large-scale LP problems in practice.

Therefore, it is generally agreed that the simplex method is not considered a practical approach for solving large-scale LP problems.

## Interior-Point Method

The interior-point method (IPM) is generally considered a favorable approach for solving large-scale linear programming (LP) problems. Theoretical analyses suggest that the computational complexity of IPM is polynomial, which makes it an attractive option.

In practical tests, it has been observed that for LP problems with dimensions on the order of $10^4$, IPM is significantly faster than the simplex method. Additionally, IPM typically requires a relatively small number of iterations to converge to an approximate solution with high accuracy, often around 20 steps.
Furthermore, the fact that commercial software like Gurobi employs IPM to solve large-scale LP problems indicates its practicality and effectiveness in real-world applications. This suggests that IPM may indeed be the future of LP-solving methods.

It is important to acknowledge that solving linear systems in interior-point methods (IPM) can become more challenging as the problem size increases. While IPM performs well for medium-sized linear programming (LP) problems, larger-scale problems may require specialized techniques to efficiently handle the linear system solving step.

Indeed, for large-scale linear programming (LP) problems, solving linear systems remains a challenging task, even with the utilization of advanced technologies such as GPUs. One of the primary challenges stems from the dynamic nature of the left-hand side (LHS) matrix in the linear systems during the iterative update steps of interior-point methods (IPM).
Due to the changing LHS matrix, it becomes necessary to solve the LP at each iteration step precisely once. This means that we cannot use any presolve technique to reduce the computation of solving linear systems.

Hence, just for me, it is still questionable whether we can use ipm to solve large-scale LP.

## ALM and ADMM

As we mentioned before, we want to use first-order methods to solve LP.
A natural idea is proximal point algorithm (PPA) and its variants. 
To introduce the ALM, we first define the augmented Lagrangian function $\mathcal{L}$ given as follows:
$$
\mathcal{L} (x, y; \rho) := \langle c, x \rangle + \langle y, Ax - b \rangle + \frac{\rho}{2} \| Ax - b \|^2.
$$
The ALM method, which has been widely studied (see [wiki](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method)), is given as:
$$
\begin{dcases}
x^{k+1} := \argmin_{x \geq 0} \mathcal{L} (x, y^k; \rho) ; \\\ 
y^{k+1} := y^k + \rho * (A x^{k+1} - b).
\end{dcases}
$$

When applying the Augmented Lagrangian Method (ALM) to solve Linear Programming (LP) problems, several challenges arise. One significant obstacle is the lack of an explicit formulation for calculating the next iteration, $ x^{k+1} $. This computation of $ x^{k+1} $ poses a major difficulty in the process.

Various methods exist to address this issue, such as Newton's method or Semi-Newton's method. However, these methods come with drawbacks. For instance, fast algorithms like the Semi-Newton method often necessitate solving linear systems, which can be problematic for large-scale LP scenarios. Additionally, regardless of the approach used to solve the subproblem, these methods typically involve iterative processes, effectively turning the application of ALM into a two-loop method.

These inherent drawbacks can render the practical application of ALM challenging. Despite these limitations, some studies have successfully utilized ALM-type algorithms to tackle LP problems, such as this [work](https://arxiv.org/abs/1903.09546), achieving promising results. (I don't know why.)

ADMM algorithm is another algorithm used to solve linear programming problems. A survey of this application can be found at: [ADMM application](https://web.stanford.edu/class/msande310/ADMM1.pdf) (seems like a homework).
In comparison to ALM, the ADMM iteration avoids the necessity of a two-loop update, although it still requires solving a linear system at each iteration step.
Moreover, as observed, the coefficient matrix of the linear system (i.e., the left-hand side part) remains fixed during the iteration.
Therefore, matrix decomposition methods can be utilized to preprocess the linear system and boost the ADMM update process.

However, numerical experiments reveal that ADMM is slower than ALM, despite ALM being a two-loop method. This observation aligns with findings from previous studies, although I'm unsure if this holds true for large-scale linear programming (LP) problems.
Furthermore, the convergence rate of ADMM is unsatisfactory.
In addition to these drawbacks, selecting the optimal hyperparameter in the ADMM update is a challenging task. Even widely-accepted empirical approaches like [ADMM Chapter 3](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) have failed to deliver in large-scale LP scenarios (at least in my experience, although I'm uncertain if this is a universal phenomenon)

In summary, though ADMM looks like better than ALM, maybe ALM is a better choice for solving LP than ADMM.

## PDHG