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
* $\sigma$-algebra of measurable subsets of $\Omega$ : $B(\Omega)$
* Probability measure $P$ on  $B(\Omega)$

Note that when the sample space $\Omega$ is finite, then $B(\Omega)=$ all subsets of $\Omega$ and the probability measure $P$ is the probability mass function.
In finite MDP, we choose the sample space $\Omega$ as
$$\Omega = \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{A} \times \mathcal{S} = (\mathcal{S} \times \mathcal{A})^{N-1} \times \mathcal{S}$$
and the event $\omega \in \Omega$ as 
$$\omega = (s_1,a_1,...,a_{N-1},s_{N-1})$$
where we refer $w$ as sample path.

Now, We define the random variables $X$, and $Y$, which take values in $\mathcal{S}$ and $\mathcal{A}$, respectively, by

$$X_t(\omega) = s_t,~Y_t(\omega)=a_t$$
