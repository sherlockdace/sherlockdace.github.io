---
title: "Quantization in LLMs"
date: 2024-10-03T20:05:43+08:00
draft: true
math: true
---

As large language models (LLMs) continue to expand in size and intricacy, mitigating their computational and energy demands has emerged as a pivotal challenge. Quantization has surfaced as a prominent approach, involving the reduction of parameter precision in models to lower bit formats. For instance, numerous methodologies truncate parameters from the conventional 16-bit floating point (FP16) or 32-bit floating point (FP32) to diminished bit formats such as 8-bit or 4-bit.

In this article, our focus will be on particular quantization models, delving into the fundamental motivations and concepts underpinning these endeavors.

## 1-BIT

Let us start from a naive model: 1-bit LLM model. The concept of 1-bit estimation, as proposed in the [1-bit LLM paper](https://arxiv.org/pdf/2310.11453), introduces a novel approach towards quantization in LLMs.

In traditional LLM models, a fundamental operation involves the linear operator `nn.Linear`. In essence, a linear operator $f$ transforms an input $X$ into an output $Y$ through the operation $f(X) := WX$, where $W$ represents a tensor. Typically, in a generic linear operator, the entries of the weight tensor $W$ are real-valued numbers. However, in the context of 1-bit quantization models, the entries are constrained to the set $\{ -1, 0, 1 \}$, significantly reducing the computational cost of the operation $f(X)$ for LLMs. This reduction in computational cost is a key advantage of the 1-bit quantization scheme, offering potential efficiency gains in the processing of Large Language Models.

![1-bit quantization linear operator vs FP16 linear operator](https://github.com/sherlockdace/sherlockdace.github.io/blob/main/content/imgs/llm_quant/llm_quant_1.jpg?raw=true)

For the 1-bit model, the input $X$ and the matrix $W$ are both transferred into the integer forms. Specifically, the matrix $W$ is modified according to the following rule:
$$
scale_w = \frac{1}{mn \sum_{i, j} |W_{ij}|} ,
$$
$$
W_q = clamp_{[-1, 1]} (round(W * scale_w)),
$$
$$
W_{dequantized} = W_q * scale_w.
$$

On the other side, the input $X$ is also required to be modified into integer numbers. The quantization formula is 
$$
scale_x = \frac{127}{|X|_{max, dim=-1}} ,
$$

$$
X_q = clamp_{[-128, 127]} (round (X * scale_x)),
$$
$$
X_{dequantized} = X_q * sacle_x.
$$

Based on this idea, the python code implementation of the 1-bit model is given as follows.
Let $LN (x):= \frac{x - \mathbb{E} (x)}{\sqrt{Var(x)}}$ be the normalization of the input $X$.
Then, we have 
```python
# Adapted from https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
import torch
import torch.nn as nn 
import torch.nn.functional as F

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
 
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinear(nn.Linear):
    """
    Only for training
    """
    def forward(self, x):
        w = self.weight
        x_norm = LN(x)
        
        # A trick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        # Perform quantized linear transformation
        y = F.linear(x_quant, w_quant)
        return y
```

Here, `BitLinear` is the low-bit estimation for the linear operator $f$. The main obstacle to training in ternary precision is that the weight values are discretized (via the round() function) and thus non-differentiable. BitLinear solves this with a nice trick [STE](https://arxiv.org/abs/1903.05662).

Hence, the architecture of the BitNet (low bit LLM model) is given as follows. 
![The architecture of BitNet, consisting of the
stacks of attentions and FFNs, where matrix multiplication is implemented as `BitLinear`](https://github.com/sherlockdace/sherlockdace.github.io/blob/main/content/imgs/llm_quant/llm_quant_2.jpg?raw=true)