---
title: "Deep Learning Theory3"
mathjax: true
layout: post
categories: media
header-includes:
  - \usepackage{amsmath}    # for advanced math environments
  - \usepackage{amsfonts}   # for math fonts
  - \usepackage{amssymb}    # for math symbols
  - \usepackage{amsthm}     # for theorem environments
  - \usepackage{bm}         # for bold symbols
  - \usepackage{mathtools}  # for math tools and extensions
  - \usepackage{mathrsfs}   # for math script font
output: pdf_document
---

## Day3
### Notations

The \( l^{(th)} \) layer's \( t^{(th)} \) preactivation component where each layer's width is \( n_l \rightarrow i \in [n_l] \):

$$
\hat{z}_i^{(l)}(x)
$$

### Neural Networks 101

For the first layer:

$$
\hat{z}_i^{(1)}(x) = b_i^{(1)} + \sum_{j=1}^{n_0} W_{ij}^{(1)} x_j \quad \text{for } i = 1, \ldots, n_1,
$$

For layers \( \ell = 1, \ldots, L - 1 \):

$$
\hat{z}_i^{(\ell+1)}(x) = b_i^{(\ell+1)} + \sum_{j=1}^{n_\ell} W_{ij}^{(\ell+1)} \sigma\left(\hat{z}_j^{(\ell)}(x)\right) \quad \text{for } i = 1, \ldots, n_{\ell+1};
$$

The output is given by:

$$
\hat{z}_{i;\delta} = \hat{z}_i^{(L)}(x_\delta)
$$

Note that \( \hat{\cdot} \) means preactivations.
Biases and weights (model parameters) are independently (& symmetrically) distributed with variances:

$$
\mathbb{E}\left[ b_i^{(\ell)} b_i^{(\ell)} \right] = \delta_{i_1i_2} C_b^{(\ell)}, \quad \mathbb{E}\left[ W_{i_1j_1}^{(\ell)} W_{i_2j_2}^{(\ell)} \right] = \delta_{i_1i_2}\delta_{j_1j_2} \frac{C_w^{(\ell)}}{n_{\ell-1}}
$$

\( C^{(l)}_b, C^{(l)}_W \) are initialization hyperparameters.

### One Aside on Gradient Descent

The parameter update equation:

$$
\theta_{\mu}(t + 1) = \theta_{\mu}(t) - \eta \sum_{\nu} \lambda_{\mu\nu} \left( \sum_{\alpha} \frac{\partial \mathcal{L}}{\partial z_{j;\alpha}} \frac{dz_{j;\alpha}}{d\theta_{\nu}} \right)
$$
