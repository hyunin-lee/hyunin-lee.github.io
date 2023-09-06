---
title:  "Deep learning theoery 1.pretraining"
mathjax: true
layout: post
categories: media
---
[Go to Notion page](https://www.notion.so/Chapter1-Pretraining-67fafb97e8fc44869454ea708e59edd3)

# Chapter1. Pretraining
The chapter 1 can be grapsed as __Wide and Deep Neural Networks are goverend by nearly-Gaussian distributions__.


## Gaussian integrals. 
### single variable gaussian distribution
we define the Gaussian distributions with zero-mean with variance K as 
$$p(z)= \frac{1}{\sqrt{2 \pi K}} e^{-z^2 / 2K}$$

We define the expectatin value of general functions $\mathcal{O}(z)$ which is observavle when $z$ follow the gaussian distributions

$$ \mathbb{E} [\mathcal{O}(z)] = \frac{1}{\sqrt{2 \pi K}}\int_{\infty}^{\infty} dz e^{-z^2 / 2K} \mathcal{O} (z)$$

### Multivariable gaussian distribution

$$p(z)= \frac{1}{\sqrt{|2 \pi K|}} \exp \left[ -\frac{1}{2} \sum_{\mu,\nu =1}^{N} z_{\mu} (K^{-1})_{\mu \nu} z_{\nu}  \right] $$






