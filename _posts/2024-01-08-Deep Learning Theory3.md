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

Taylor expansion:

$$
\begin{aligned}
    \hat{z}_{i;\delta}(t + 1) &= \hat{z}_{i;\delta}(t) \\
    &- \eta \sum_{j,\alpha} \left( \sum_{\mu,\nu} \lambda_{\mu\nu} \frac{dz_{i;\delta}}{d\theta_{\mu}} \frac{dz_{j;\alpha}}{d\theta_{\nu}} \right) \frac{\partial \mathcal{L}}{\partial z_{j;\alpha}} \quad \text{(NTK)} \\
    &+ \frac{\eta^2}{2} \sum_{j_1,j_2,\alpha_1,\alpha_2} \left( \sum_{\mu_1,\mu_2,\nu_1,\nu_2} \lambda_{\mu_1\nu_1} \lambda_{\mu_2\nu_2} \frac{d^2 z_{i;\delta}}{d\theta_{\mu_1}d\theta_{\mu_2}} \frac{dz_{j_1;\alpha_1}}{d\theta_{\nu_1}} \frac{dz_{j_2;\alpha_2}}{d\theta_{\nu_2}} \right) \frac{\partial \mathcal{L}}{\partial z_{j_1;\alpha_1}} \frac{\partial \mathcal{L}}{\partial z_{j_2;\alpha_2}} \quad \text{(dNTK)} \\
    &- \frac{\eta^3}{6} \sum_{j_1,j_2,j_3,\alpha_1,\alpha_2,\alpha_3} \left( \sum_{\mu_1,\mu_2,\mu_3,\nu_1,\nu_2,\nu_3} \lambda_{\mu_1\nu_1} \lambda_{\mu_2\nu_2} \lambda_{\mu_3\nu_3} \frac{d^3 z_{i;\delta}}{d\theta_{\mu_1}d\theta_{\mu_2}d\theta_{\mu_3}} \frac{dz_{j_1;\alpha_1}}{d\theta_{\nu_1}} \frac{dz_{j_2;\alpha_2}}{d\theta_{\nu_2}} \frac{dz_{j_3;\alpha_3}}{d\theta_{\nu_3}} \right) \\
    &\quad \frac{\partial \mathcal{L}}{\partial z_{j_1;\alpha_1}} \frac{\partial \mathcal{L}}{\partial z_{j_2;\alpha_2}} \frac{\partial \mathcal{L}}{\partial z_{j_3;\alpha_3}} + \dots
\end{aligned}
$$

### Neural Tangent Kernel (NTK)

The Neural Tangent Kernel (NTK) \( H(t) \) and its differential \( dH(t) \):

$$
\hat{H}^{(\ell)}_{i_1i_2;\delta_1\delta_2} \equiv \sum_{\mu, \nu} \lambda_{\mu\nu} \frac{d\hat{z}^{(\ell)}_{i_1;\delta_1}}{d\theta_{\mu}} \frac{d\hat{z}^{(\ell)}_{i_2;\delta_2}}{d\theta_{\nu}}, \quad \{ \theta_{\mu} \} = \{ b^{(\ell)}_i, W^{(\ell)}_{ij} \}
$$

$$
\hat{H}_{i_1i_2;\delta_1\delta_2} = \hat{H}^{(L)}_{i_1i_2;\delta_1\delta_2}
$$

Diagonal, group-by-group, learning rate:

$$
\lambda^{b(\ell)}_{i_1 i_2} = \delta_{i_1i_2} \lambda^{(\ell)}_b, \quad \lambda^{W(\ell)}_{i_1j_1 i_2j_2} = \delta_{i_1i_2} \delta_{j_1j_2} \frac{\lambda^{(\ell)}_W}{n_{\ell-1}}
$$

### Two Pedagogical Simplifications

[See "PDLT" (arXiv:2106.10165) for more general cases.]

1. Single input; drop sample indices:

   $$
   x_{j;\delta} \rightarrow x_j, \quad \hat{z}^{(\ell)}_{j;\delta} \rightarrow \hat{z}^{(\ell)}_j, \quad \hat{H}^{(\ell)}_{i_1i_2;\delta_1\delta_2} \rightarrow \hat{H}^{(\ell)}_{i_1i_2}
   $$

2. Layer-independent hyperparameters; drop layer indices from them:

   $$
   C^{(\ell)}_b = C_b, \quad C^{(\ell)}_W = C_W, \quad \lambda^{(\ell)}_b = \lambda_b, \quad \lambda^{(\ell)}_W = \lambda_W
   $$
