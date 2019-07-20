
# E_M [f(x)]
def monte_carlo(function, model, batch_size=1024):
    return function(model.sample(batch_size)).mean()


def expectation(model, batch_size=1024):
    return model.sample(batch_size).mean()


def variance(model, batch_size=1024):
    samples = model.sample(batch_size)
    return (samples - samples.mean()).pow(2).mean()


def median(model, batch_size=1024):
    return model.sample(batch_size).median()


def cdf(model, c, batch_size=1024):
    # look into clamp for Differentiable?
    return (model.sample(batch_size) <= c).sum().float().div(batch_size)


# EOF
