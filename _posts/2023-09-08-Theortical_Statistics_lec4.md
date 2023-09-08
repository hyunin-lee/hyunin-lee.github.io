---
title:  "Formal model in stochastic process by Markov Decision Process"
mathjax: true
layout: post
categories: media
header-includes:
  \usepackage{amsmath}
  \usepackage{amssymb}
  \usepackage{amsthm}
---

## Motivation: Coin flipping 

Suppose $X_{1},X_{2},...,X_{n} \sim \text{Bernoulli}({\theta})$.  
Then $X \sim \Pi_{i} \theta^{X_i} (1- \theta^{(1- X_{i})})$ on $[0,1]^{n}$
