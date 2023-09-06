---
title:  "Formal model in stochastic process by Markov Decision Process"
mathjax: true
layout: post
categories: media
---

_The contents are from [Markov Decision Processes: Discrete Stochastic Dynamic Programming - MARTIN L. PUTERMAN], section 2.1.6_


# Formal model for stochastic process in MDP

The probability model consists of three elements: 
* A sample space $\Omega$
* $\simga$-algebra of measurable subsets of $\Omega$ : $B(\Omega)$
* Probability measure $P$ on  $B(\Omega)$

Note that when the sample space $\Omega$ is finite, then $B(\Omega)=$ all subsets of $\Omega$ and the probabiltiy measure $P$ is probability mass function.
In finite MDP, we choose 
$$\Omega = \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{A} \times \mathcal{S} = (\mathcal{S} \times \mathcal{A})^{N-1} \times \mathcal{S}$$
