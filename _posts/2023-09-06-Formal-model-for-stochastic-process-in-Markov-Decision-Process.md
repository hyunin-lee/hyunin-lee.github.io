---
title: "Formal model in stochastic process by Markov Decision Process"
layout: post
categories: media
output: pdf_document
mathjax: true
---

_The contents are from [Markov Decision Processes: Discrete Stochastic Dynamic Programming - MARTIN L. PUTERMAN], section 2.1.6_

# Probability model for stochastic process in MDP

The probability model consists of three elements: 
* A sample space \( \Omega \)
* \(\sigma\)-algebra of measurable subsets of \( \Omega \): \( B(\Omega) \)
* Probability measure \( P \) on  \( B(\Omega) \)

Note that when the sample space \( \Omega \) is finite, then \( B(\Omega) \) equals all subsets of \( \Omega \) and the probability measure \( P \) is the probability mass function.

In finite MDP, we choose the sample space \( \Omega \) as
\[\Omega = \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{A} \times \mathcal{S} = (\mathcal{S} \times \mathcal{A})^{N-1} \times \mathcal{S}\]
and the event \( \omega \in \Omega \) as 
\[\omega = (s_1, a_1, \ldots, a_{N-1}, s_{N-1})\]
where we refer \( w \) as sample path.

Also, we define the random variables \( X \), and \( Y \), which take values in \( \mathcal{S} \) and \( \mathcal{A} \), respectively, by

\[ X_t(\omega) = s_t, \quad Y_t(\omega) = a_t \]

and the history process \( Z_t \) as 

\[ Z_1(w) = s_1, \quad Z_t(w) = (s_1, a_1, \ldots, s_t) \].

Now, a randomized history-dependent policy \( \pi = (d_1, d_2, \ldots , d_{N-1}), \quad N \leq \infty \) induces a probability \( P^{\pi} \) on \( (\Omega, B(\Omega)) \) through 

\[
\begin{aligned}
  & P^{\pi}(X_t = s) = P_t(s), \\ 
  & P^{\pi}(Y_t = a \mid Z_t = h_t) = q_{d_t(h_t)}(a),\\ 
  & P^{\pi}(X_{t+1}=s \mid Z_t=(h_{t-1}, a_{t-1}, s_{t}), Y_t = a_t) = p_t(s \mid s_t, a_t)
\end{aligned}
\]

so that the probability of a sample path \( \boldsymbol{\omega} = (s_1, a_1, \ldots, s_N) \) is given as 

\[ P^{\pi}(s_1, a_1, \ldots, s_N) = P_1(s_1) q_{d_1(s_1)}(a_1) p_1(s_2 \mid s_1, a_1) q_{d_2(h_2)}(a_2) \ldots q_{d_{N-1}(h_{N-1})}(a_{N-1}) p_{N-1}(s_N) \]



