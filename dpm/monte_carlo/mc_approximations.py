
# E_M [f(x)]
def monte_carlo(function, model, batch_size=1024):
    return function(model.sample(batch_size)).mean()


def expectation(model, batch_size=1024):
    return model.sample(batch_size).mean()


def variance(model, batch_size=1024):
    samples = model.sample(batch_size)
    return (samples - samples.mean()).pow(2).mean()


def standard_deviation(model, batch_size=1024):
    return variance(model, batch_size).sqrt()


def skewness(model, batch_size=1024):
    samples = model.sample(batch_size)
    return ((samples - samples.mean()) / samples.std()).pow(3).mean()


def kurtosis(model, batch_size=1024):
    samples = model.sample(batch_size)
    return ((samples - samples.mean()) / samples.std()).pow(4).mean() - 3.


def median(model, batch_size=1024):
    return model.sample(batch_size).median()


def cdf(model, c, batch_size=1024):
    return (model.sample(batch_size) <= c).sum().float().div(batch_size)


def entropy(model, batch_size=1024):
    return -monte_carlo(model.log_prob, model, batch_size)


def max(model, batch_size=1024):
    return model.sample(batch_size).max()


def min(model, batch_size=1024):
    return model.sample(batch_size).min()
