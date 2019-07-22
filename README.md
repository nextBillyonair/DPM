# DPM
Differentiable Probabilistic Models

---

## TODO:
  * Finish VAE example in Notebook
  * GAN Example using Distribution Components
  * GLM Models from TensorFlow
  * Add more Transforms from TensorFlow

---

## Table of Contents
1. [Distributions](#distributions)
1. [Transforms](#transforms)
1. [Mixture Models](#mixture-models)
1. [Divergences](#divergences)
1. [Earth Mover's Distance](#emd)
1. [Adversarial Loss](#adversarial-loss)
1. [ELBO](#elbo)
1. [Monte Carlo Approximations](#monte_carlo_approximation)
1. [Monte Carlo Sampling](#monte_carlo_sampling)
1. [MCMC Methods](#mcmc-methods)
1. [Notes](#notes)

---

# Distributions <a name="distributions"></a>
  1. [Normal (Multivariate)](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
  1. [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
  1. [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)
  1. [Beta](https://en.wikipedia.org/wiki/Beta_distribution)
  1. [Log Normal](https://en.wikipedia.org/wiki/Log-normal_distribution)
  1. [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
  1. [Relaxed Bernoulli](https://arxiv.org/abs/1611.00712)
  1. [Gumbel Softmax (Relaxed Categorical)](https://arxiv.org/abs/1611.01144)
  1. [Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
  1. [Student T](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
  1. [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
  1. [Fisher-Snedecor (F-Distribution)](https://en.wikipedia.org/wiki/F-distribution)
  1. [Dirac Delta](https://en.wikipedia.org/wiki/Dirac_delta_function)
  1. [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution)
  1. Half Cauchy
  1. Half Normal
  1. Data Distribution
      * Randomly sample from a given set of data.
  1. Conditional Model
      * Uses a Neural Network to take inputs and create the parameters of a distribution.
      * Sampling -> takes a value, runs the network to create the distribution,
        sample from conditional distribution.
      * Log Prob -> Create distribution conditioned on X, take log_prob w.r.t. Z
  1. Transform Distribution
      * Composes a list of [transforms](#transforms) on a distribution
      * Example: Exp(Normal) ~ LogNormal
  1. [Convolution](https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions)
      * Sum of component distributions, only allows sampling
  1. [ChiSquare](https://en.wikipedia.org/wiki/Chi-squared_distribution)
  1. [Logistic](https://en.wikipedia.org/wiki/Logistic_distribution)
  1. Generator
      * Uses a latent distribution to sample inputs to a neural network to
      generate a distribution. Train with the adversarial losses.


# Transforms <a name="transforms"></a>
  1. Exp
  1. Log
  1. Power
  1. Reciprocal
  1. Square
  1. Sigmoid
  1. Logit
  1. Affine
  1. Expm1
  1. Gumbel
  1. SinhArcsinh
  1. Softplus
  1. Softsign
  1. Tanh
  1. InverseTransform (Inverts a transform)

# Mixture Models <a name="mixture-models"></a>
  1. [Mixture Model](https://en.wikipedia.org/wiki/Mixture_model)
      * Static weights to pick from sub-models using a categorical distribution.
  1. [Gumbel Mixture Model](https://arxiv.org/abs/1611.01144)
      * Uses the Gumbel Softmax as a differentiable approximation to the
      categorical distribution, allowing mixture weights to be learned.
  1. Infinite Mixture Model
      * Student T written as a Mixture Model.

# Divergences <a name="divergences"></a>
  1. [Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)
  1. [Perplexity](https://en.wikipedia.org/wiki/Perplexity)
  1. [Forward KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
      * P Model -> Sampling (rsample)
      * Q Model -> PDF Function (log_prob)
  1. [Reverse KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
      * P Model -> PDF Function (log_prob)
      * Q Model -> Sampling + PDF Function
  1. [Jensen-Shannon Divergence (JS)](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)
      * P Model -> PDF + Sampling
      * Q Model -> PDF + Sampling
  1. In Progress:
      * Total Variation
      * Pearson
      * F-Divergence
      * Quadrature Methods for F-Divergence

# [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) for Discrete Distributions <a name="emd"></a>
   1. Linear Programming solution to 2 discrete distributions (histograms)
   1. Primal and Dual Formulation

# Adversarial Loss <a name="adversarial-loss"></a>
  1. [Adversarial Loss (aka GAN Loss)](https://arxiv.org/pdf/1711.10337.pdf)
      * Hides a discriminator under the loss function, and computes the adversarial loss
      * P Model -> Sampling (rsample)
      * Q Model -> Sampling (rsample)
      * Must use RMSprop
  1. [GAN Loss](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
  1. [MMGAN Loss](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
  1. [WGAN Loss](https://arxiv.org/pdf/1701.07875.pdf)
  1. [LSGAN Loss](https://arxiv.org/pdf/1611.04076.pdf)

# ELBO <a name="elbo"></a>
  1. Implements SVI with ELBO loss.
  1. Requires a Conditional Model to learn, in addition to P and Q models.

# Monte Carlo Approximations <a name="monte_carlo_approximation"></a>
  1. [Monte Carlo Approximation:](https://en.wikipedia.org/wiki/Monte_Carlo_method) Expectation of F(X) wrt X ~ Model
      * x_i sampled from Model, then averages F(x_i), see below for specific examples
  1. Expectation -> average of samples
  1. Variance -> average of squared difference between samples and mean
  1. Median -> median of samples
  1. CDF -> proportion of samples <= c
  1. Entropy -> average negative log_prob of samples
  1. Max -> approximate Maximum limit of model
  1. Min -> approximate Minimum limit of model

# Monte Carlo Sampling <a name="monte_carlo_sampling"></a>
  1. [Rejection Sampling](https://en.wikipedia.org/wiki/Rejection_sampling) -> given a model, proposal model, and M, attempt to create samples similar to model via sampling the proposal model.
  1. Box Muller -> Generates a Standard Normal
  1. Marsaglia-Bray -> Generates a Standard Normal
  1. TESTING:
      1. Beta Sampling
      1. Double Exponential Sampling

# MCMC Methods <a name="mcmc-methods"></a>
  1. [Metropolis–Hastings](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm)
      * MCMC sampling method to generate samples from a unknown distribution
        * Requires distribution to have a log_prob method implemented.

# Notes <a name="notes"></a>
  * Sampling must be done through a reparameterized version of the
    distribution to allow gradients to back-prop through samples.
  * Probabilities are in log form, for numerical stability.
