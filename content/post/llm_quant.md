---
title: "Quantization in LLMs"
date: 2024-10-03T20:05:43+08:00
draft: true
math: true
---

As large language models (LLMs) continue to expand in size and intricacy, mitigating their computational and energy demands has emerged as a pivotal challenge. Quantization has surfaced as a prominent approach, involving the reduction of parameter precision in models to lower bit formats. For instance, numerous methodologies truncate parameters from the conventional 16-bit floating point (FP16) or 32-bit floating point (FP32) to diminished bit formats such as 8-bit or 4-bit.

In this article, our focus will be on particular quantization models, delving into the fundamental motivations and concepts underpinning these endeavors.

## 1-BIT

We start from the naive idea: 1-bit estimation proposed in [1-bit LLM](https://arxiv.org/pdf/2310.11453).
One of the basic operator in LLM models is the linear operator `nn.Linear`. 
Roughly speaking, a linear operator $f$ maps the input $X$ into the output $Y$ with $f(X) := WX$ where $W$ is a tensor. 
For the generic linear operator, the entries of $W$ are float numbers, while the entries of the 1-bit quantization model are in $\{ -1, 0, 1 \}$.

![1-bit quantization linear operator vs FP16 linear operator]("https://github.com/sherlockdace/sherlockdace.github.io/blob/main/content/imgs/llm_quant/llm_quant_1.jpg")

