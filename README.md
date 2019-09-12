# DPM
Differentiable Probabilistic Models

---

## Table of Contents
1. [Distributions](#distributions)
1. [Transforms](#transforms)
1. [Criterion](#criterion)
1. [Models](#models)
1. [Monte Carlo](#monte_carlo)

---

# Distributions <a name="distributions"></a>
  1. [Arcsine](https://en.wikipedia.org/wiki/Arcsine_distribution)
  1. [Asymmetric Laplace](https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution)
  1. [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)
  1. [Beta](https://en.wikipedia.org/wiki/Beta_distribution)
  1. [Categorical](https://en.wikipedia.org/wiki/Categorical_distribution)
  1. [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)
  1. [ChiSquare](https://en.wikipedia.org/wiki/Chi-squared_distribution)
  1. Conditional Model
      * Uses a Neural Network to take inputs and create the parameters of a distribution.
      * Sampling -> takes a value, runs the network to create the distribution,
        sample from conditional distribution.
      * Log Prob -> Create distribution conditioned on X, take log_prob w.r.t. Z
  1. [Convolution](https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions) -Sum of component distributions, only allows sampling
  1. Data Distribution - Randomly sample from a given set of data.
  1. [Dirac Delta](https://en.wikipedia.org/wiki/Dirac_delta_function)
  1. [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
  1. [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
  1. [Fisher-Snedecor (F-Distribution)](https://en.wikipedia.org/wiki/F-distribution)
  1. [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
  1. Generator
      * Uses a latent distribution to sample inputs to a neural network to
      generate a distribution. Train with the adversarial losses.
  1. [Gumbel Softmax (Relaxed Categorical)](https://arxiv.org/abs/1611.01144)
  1. [Gumbel](https://en.wikipedia.org/wiki/Gumbel_distribution)
  1. Half Cauchy
  1. Half Normal
  1. [Hyperbolic Secant](https://en.wikipedia.org/wiki/Hyperbolic_secant_distribution)
  1. [Kumaraswamy](https://en.wikipedia.org/wiki/Kumaraswamy_distribution)
  1. [Langevin](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)
      * Adds Langevin Dynamics to sampling methods (see wikipedia)
  1. [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution)
  1. [Log Cauchy](https://en.wikipedia.org/wiki/Log-Cauchy_distribution)
  1. [Log Laplace](https://en.wikipedia.org/wiki/Log-Laplace_distribution)
  1. [Log Normal](https://en.wikipedia.org/wiki/Log-normal_distribution)  
  1. [Logistic](https://en.wikipedia.org/wiki/Logistic_distribution)
  1. [Logit Normal](https://en.wikipedia.org/wiki/Logit-normal_distribution)
  1. [Normal (Multivariate)](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
  1. [Pareto](https://en.wikipedia.org/wiki/Pareto_distribution)
  1. [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution)
  1. [Rayleigh](https://en.wikipedia.org/wiki/Rayleigh_distribution)
  1. [Relaxed Bernoulli](https://arxiv.org/abs/1611.00712)
  1. [Student T](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
  1. Transform Distribution
      * Composes a list of [transforms](#transforms) on a distribution
      * Example: Exp(Normal) ~ LogNormal
  1. [Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
  1. [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution)
  1. [Mixture Model](https://en.wikipedia.org/wiki/Mixture_model)
      * Static weights to pick from sub-models using a categorical distribution.
  1. [Gumbel Mixture Model](https://arxiv.org/abs/1611.01144)
      * Uses the Gumbel Softmax as a differentiable approximation to the
      categorical distribution, allowing mixture weights to be learned.
  1. Infinite Mixture Model
      * Student T written as a Mixture Model.

# Transforms <a name="transforms"></a>
  1. Affine
  1. Exp
  1. Expm1
  1. Gumbel
  1. Identity
  1. InverseTransform (Inverts a transform)
  1. Kumaraswamy
  1. Log
  1. Logit
  1. Planar
  1. Power
  1. Radial
  1. Reciprocal
  1. Sigmoid
  1. SinhArcsinh
  1. Softplus
  1. Softsign
  1. Square
  1. Tanh
  1. Weibull

# Criterion <a name="criterion"></a>
  1. Divergences
      1. [Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)
      1. [Perplexity](https://en.wikipedia.org/wiki/Perplexity)
      1. Exponential
      1. [Forward KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
          * P Model -> Sampling (rsample)
          * Q Model -> PDF Function (log_prob)
      1. [Reverse KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
          * P Model -> PDF Function (log_prob)
          * Q Model -> Sampling + PDF Function
      1. [Jensen-Shannon Divergence (JS)](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)
          * P Model -> PDF + Sampling
          * Q Model -> PDF + Sampling
  1. Adversarial
      1. [GAN Loss](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
      1. [MMGAN Loss](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
      1. [WGAN Loss](https://arxiv.org/pdf/1701.07875.pdf)
      1. [LSGAN Loss](https://arxiv.org/pdf/1611.04076.pdf)
  1. Variational
      1. ELBO
          * Implements SVI with ELBO loss.
          * Requires a Conditional Model to learn, in addition to P and Q models.

# Models <a name="models"></a>
  1. Regression
      1. Linear Regression (Normal)
      1. L1 Regression (Laplace)
      1. Ridge Regression (Normal + Normal Prior on weights) (Bayesian Linear Regression)
      1. Lasso Regression (Normal + Laplace Prior on weights)
  1. Classification
      1. Logistic Regression (Bernoulli)
      1. Bayesian Logistic Regression (Bernoulli + Normal Prior on weights)
      1. Softmax Regression (Categorical)
      1. Gaussian Naive Bayes
      1. Bernoulli Naive Bayes
      1. Multinomial Naive Bayes
      1. Linear Discriminant Analysis (Shared Covariance)
      1. Gaussian Discriminant Analysis (Multivariate Normal)
  1. Clustering
      1. Gaussian Mixture Model
  1. Decomposition
      1. Functional PCA
      1. Dynamic SVD Based (can update projection size)
      1. EM PPCA
      1. Variational PPCA
  1. Unconstrained Matrix Factorization
  1. Generative Adversarial Networks
      1. GAN
      1. MMGAN
      1. WGAN
      1. LSGAN

# Monte Carlo <a name="monte_carlo"></a>
  1. Approximations (Integration, Expectation, Variance, etc.)
  1. Inverse Transform Sampling
  1. Rejection Sampling (and Mode Sampling)
  1. Metropolis
  1. Metropolis-Hastings
  1. Metropolis-Adjusted Langevin Algorithm (MALA)
  1. Hamiltonian Monte Carlo (HMC)
