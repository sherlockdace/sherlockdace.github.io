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
x^{k+1} := \argmin_x \mathcal{L} (x, y^k; \rho) = (A^\top A)^{-1} \left(A^\top b - \frac{1}{\rho} (c + A^\top y^k) \right); \\\ 
y^{k+1} := y^k + \rho * (A x^{k+1} - b).
\end{dcases}
$$

In general, we employ matrix decomposition, like LU decomposition or Cholesky decomposition, to matrix $A^\top A$. This decomposition is then used to solve linear systems arising in the update steps more efficiently.

However, despite these advantages, the ALM still has some practical limitations. It's important to note that even in ALM, we need to solve a linear system at each step to obtain the update for $x^{k+1}$.
What's worse is that the matrix $A^\top A$ is not always an ideal matrix. In fact, it tends to be ill-conditioned for large-scale linear programming (LP) problems, which means its decomposition can be numerically unstable.

To address this issue and improve the stability of the ALM update, we can utilize the [Proximal ALM](https://pubsonline.informs.org/doi/10.1287/moor.1.2.97). In this approach, instead of decomposing the matrix $A^\top A$, we decompose the matrix $\mu I + A^\top A$, where $\mu$ is a positive constant.
Admittedly, the Proximal ALM may be slightly slower than the regular ALM in practice. However, by introducing the additional term $\mu I$, the matrix decomposition becomes more feasible and implementable, thus enhancing the overall stability of the method.

Besides the ALM and Proximal ALM, another algorithm called ADMM is also applied to solve LP.
Some recent works, like [new ADMM](https://dl.acm.org/doi/pdf/10.5555/3294771.3294912) (not a new one in fact), [ADMM application](https://web.stanford.edu/class/msande310/ADMM1.pdf) (looks like a homework), [enhanced ADMM](https://arxiv.org/abs/2209.01793) (ADMM + ipm, 😓), studied the ADMM and its variants to efficiently solve LP.
While recent works have explored different aspects of ADMM, such as its applications and enhancements, it is important to note that ADMM and its variants still require solving linear systems at each iteration.

But let's not forget about our ALM buddies. Sure, they may be slower than the flashy ipm crowd, but they have their reasons (or maybe they're just a little slower in general). These ALM types also need to solve linear systems at each step, which can be a real headache for large-scale LP problems. So, why do we use ALM type algorithms instead of IPM? 
I don't know. But the fact is that the ALM type algorithms are actually employed in real world applications. 

## PDHG