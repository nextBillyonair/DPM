import torch.optim as optim

from mixture_model import MixtureModel, GumbelMixtureModel

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


def train(p_model, q_model, method, epochs=1000, batch_size=64,
          lr=0.01, optimizer='Adam', track_parameters=True):

    criterion = method(p_model.n_dims)
    optimizer = getattr(optim, optimizer)(q_model.parameters(), lr=lr)
    stats = Statistics()

    for i in range(epochs):
        loss = criterion(p_model, q_model, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats.update({'loss':loss.item()})
        if track_parameters:
            update_stats(stats, q_model)

    return stats
