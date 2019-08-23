from torch import optim
from dpm.mixture_models import MixtureModel, GumbelMixtureModel
from dpm.elbo import ELBO

class Statistics:
    def __init__(self):
        self.data = {}

    def update(self, values):
        for k, v in values.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)


def update_stats(stats, model):
    if 'get_parameters' not in dir(model):
        return
    stats.update(model.get_parameters())


def gradient_clipping(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip, clip)


def l2_regularize(model):
    loss = 0.0
    for p in model.parameters():
        loss += p.pow(2).mean()
    return loss


def l1_regularize(model):
    loss = 0.0
    for p in model.parameters():
        loss += p.abs().mean()
    return loss


def train(p_model, q_model, criterion, epochs=1000, batch_size=64,
          lr=0.01, optimizer='Adam', track_parameters=True, log_interval=None,
          stats=None, clip_gradients=None, l2_penalty=None, l1_penalty=None):

    optimizer = getattr(optim, optimizer)(q_model.parameters(), lr=lr)

    if stats is None:
        stats = Statistics()

    for epoch in range(epochs):
        optimizer.zero_grad()
        if 'zero_grad' in dir(criterion):
            criterion.zero_grad()

        loss = criterion(p_model, q_model, batch_size)

        if l2_penalty:
            loss += l2_penalty * l2_regularize(q_model)
        if l1_penalty:
            loss += l1_penalty * l1_regularize(q_model)

        loss.backward()

        if clip_gradients:
            gradient_clipping(q_model, clip_gradients)

        optimizer.step()
        if 'step' in dir(criterion):
            criterion.step()

        stats.update({'loss':loss.item()})
        if track_parameters:
            update_stats(stats, q_model)

        if log_interval is not None and epoch % log_interval == 0:
            print(f"[Epoch {epoch}/{epochs}]\tLoss {loss.item():.2f}")

    return stats
