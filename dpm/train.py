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
    if isinstance(model, MixtureModel) or isinstance(model, GumbelMixtureModel):
        # figure out later
        return
    stats.update(model.get_parameters())


def train(p_model, q_model, criterion, epochs=1000, batch_size=64,
          lr=0.01, optimizer='Adam', track_parameters=True, log_interval=None,
          stats=None):

    optimizer = getattr(optim, optimizer)(q_model.parameters(), lr=lr)

    if stats is None:
        stats = Statistics()

    for epoch in range(epochs):
        optimizer.zero_grad()
        if 'zero_grad' in dir(criterion):
            criterion.zero_grad()

        loss = criterion(p_model, q_model, batch_size)
        loss.backward()

        optimizer.step()
        if 'step' in dir(criterion):
            criterion.step()

        stats.update({'loss':loss.item()})
        if track_parameters:
            update_stats(stats, q_model)

        if log_interval is not None and epoch % log_interval == 0:
            print(f"[Epoch {epoch}/{epochs}]\tLoss {loss.item():.2f}")

    return stats
