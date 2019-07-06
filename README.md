# DPM
Differentiable Probabilistic Models

## Table of Contents
1. [Distributions](#distributions)
2. [Conditional Model](#conditional_model)
3. [Mixture Models](#mixture-models)
4. [Divergences](#divergences)
5. [Adversarial Loss](#adversarial-loss)
6. [ELBO](#elbo)
7. [MCMC Methods](#mcmc-methods)
8. [Notes](#notes)

# Distributions <a name="distributions"></a>
  1. [Normal (Multivariate)](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
  2. [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
  3. [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)
  4. [Beta](https://en.wikipedia.org/wiki/Beta_distribution)
  5. [Log Normal](https://en.wikipedia.org/wiki/Log-normal_distribution)
  6. [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
  7. [Relaxed Bernoulli](https://arxiv.org/abs/1611.00712)
  8. [Gumbel Softmax (Relaxed Categorical)](https://arxiv.org/abs/1611.01144)
  9. [Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
  10. [Student T](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
  11. [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
  12. [Fisher-Snedecor (F-Distribution)](https://en.wikipedia.org/wiki/F-distribution)
  13. [Dirac Delta](https://en.wikipedia.org/wiki/Dirac_delta_function)
  14. Data Distribution - can randomly sample from a given set of data.

# Conditional Model <a name="conditional_model"></a>
  1. Uses a Neural Network to take inputs and create the parameters of a distribution.
  2. Sampling -> takes a value, runs the network to create the distribution,
    sample from conditional distribution.
  3. Log Prob -> Create distribution conditioned on X, take log_prob w.r.t. Z

# Mixture Model <a name="mixture-model"></a>
  1. [Mixture Model](https://en.wikipedia.org/wiki/Mixture_model)
      * Static weights to pick from sub-models using a categorical distribution.
  2. [Gumbel Mixture Model](https://arxiv.org/abs/1611.01144)
      * Uses the Gumbel Softmax as a differentiable approximation to the
      categorical distribution, allowing mixture weights to be learned.
  3. Infinite Mixture Model
      * Student T written as a Mixture Model.

# Divergences <a name="divergences"></a>
  1. [Forward KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
      * P Model -> Sampling (rsample)
      * Q Model -> PDF Function (log_prob)
  2. [Reverse KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
      * P Model -> PDF Function (log_prob)
      * Q Model -> Sampling + PDF Function
  3. [Jensen-Shannon Divergence (JS)](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)
      * P Model -> PDF + Sampling
      * Q Model -> PDF + Sampling

# Adversarial Loss <a name="#adversarial-loss"></a>
  1. [Adversarial Loss (aka GAN Loss)](https://arxiv.org/pdf/1711.10337.pdf)
      * Hides a discriminator under the loss function, and computes the adversarial loss
      * P Model -> Sampling (rsample)
      * Q Model -> Sampling (rsample)
      * Must use RMSprop
      * Algorithms:
          1. [GAN Loss](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
          2. [MMGAN Loss](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
          3. [WGAN Loss](https://arxiv.org/pdf/1701.07875.pdf)
          4. [LSGAN Loss](https://arxiv.org/pdf/1611.04076.pdf)

# ELBO <a> name="#elbo"</a>
  1. Implements SVI with ELBO loss.
  2. Requires a Conditional Model to learn, in addition to P and Q models.

# MCMC Methods <a name="mcmc-methods"></a>
  1. [Metropolis–Hastings](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm)
      * MCMC sampling method to generate samples from a unknown distribution
        * Requires distribution to have a log_prob method implemented.

# Notes <a name="notes"></a>
  * Sampling must be done through a reparameterized version of the
    distribution to allow gradients to back-prop through samples.
  * Probabilities are in log form, for numerical stability.
