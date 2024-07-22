---
title: "How to Solve Linear Programming Efficiently?"
date: 2024-07-22T16:58:56+08:00
draft: true
math: true
---

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

## Simplex Method and Interior-Point Method

Theoretical analyses show that the computation complexity of simplex method is exponential, though it is extremely fast for small-scale LP problems. 
Maybe the simplex method is the fastest algorithm to solve LP when both $m$ and $n$ is not larger than $10^3$.
However, as we observed, even for not too large $m, n$, are the magnitude of $10^4$, the simplex method is much slower than the interior point.
Hence, simplex method is not a suitable method to solve the LP.

What about the interior-point method (IPM)? 
In general, ipm maybe the vanilla way to solve large-scale LP problems. 
As we know, the commercial software Gurobi uses the ipm to solve large-scale LP.
