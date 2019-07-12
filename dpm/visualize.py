import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dpm.distributions import Uniform, Normal
from torch.nn.functional import softplus
import torch
plt.style.use('seaborn-darkgrid')

def plot_stats(stats, goals=None):
    fig, axes = plt.subplots(1, len(stats.data.keys()), figsize=(12, 6), squeeze=False)
    for index, value in enumerate(stats.data.keys()):
        axes[0, index].set_title(f"{value} curve")
        axes[0, index].set_xlabel('Epoch')
        axes[0, index].set_ylabel(value)
        axes[0, index].plot(np.arange(0.0, len(stats.data[value]), 1.0),
                         stats.data[value], color="#955196")
    if goals is not None:
        for i, goal in enumerate(goals):
            axes[0, i+1].axhline(goals[i], color="#ff6e54", linewidth=4)
    plt.tight_layout()
    # plt.show()

def plot_models(p_model, q_model, batch_size=10000, n_plot=500):
    if p_model.n_dims == 1:
        return plot_models_1D(p_model, q_model, batch_size)
    return plot_models_2D(p_model, q_model, batch_size, n_plot)

def plot_models_1D(p_model, q_model, batch_size=10000):
    q_samples = q_model.sample(batch_size).detach().numpy()
    p_samples = p_model.sample(batch_size).detach().numpy()
    ax = sns.distplot(q_samples, color = '#003f5c', label="Learned Model")
    ax = sns.distplot(p_samples, color = '#ffa600', label="True Model")
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Models")
    plt.legend()
    # plt.show()


def plot_models_2D(p_model, q_model, batch_size=10000, n_plot=500):
    p_samples = p_model.sample(batch_size).detach().numpy()

    x_min, x_max, y_min, y_max = [-10.0, 10.0, -10.0, 10.0]

    plt.scatter(p_samples[:, 0], p_samples[:, 1],
                label="True Model", s=4, color="#dd5182")

    if not isinstance(p_model, Uniform):
        plot_x, plot_y = np.linspace(x_min, x_max, n_plot), np.linspace(y_min, y_max, n_plot)
        plot_x, plot_y = np.meshgrid(plot_x, plot_y)

        grid_data = torch.tensor(list(zip(plot_x.reshape(-1), plot_y.reshape(-1))))
        log_probs = q_model.log_prob(grid_data).detach().numpy()
        c2 = plt.contour(plot_x, plot_y,
                         log_probs.reshape(n_plot, n_plot),
                         levels=50, linestyles="solid", cmap="viridis")
        plt.colorbar(c2)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("P Samples vs Q Contour")
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    # plt.show()



def plot_model(model, batch_size=10000):
    if model.n_dims == 1:
        return plot_model_1D(model, batch_size)
    return plot_model_2D(model, batch_size)

def plot_model_1D(model, batch_size=10000):
    samples = model.sample(batch_size).detach().numpy()
    ax = sns.distplot(samples, color = '#003f5c', label="Model")
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Model")
    # plt.show()

def plot_model_2D(model, batch_size=10000):
    plot_models_2D(model, model, batch_size)



def plot_hists(samples_1, samples_2, labels=["Accepted Samples", "True Model"]):
    if samples_1.size(1) == 1:
        return plot_hists_1D(samples_1, samples_2, labels)
    return plot_hists_2D(samples_1, samples_2, labels)

def plot_hists_1D(samples_1, samples_2, labels=["Accepted Samples", "True Model"]):
    ax = sns.distplot(samples_1, color = '#003f5c', label=labels[0])
    ax = sns.distplot(samples_2, color = '#ffa600', label=labels[1])
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Model")
    plt.legend()
    # plt.show()

def plot_hists_2D(samples_1, samples_2, labels=["Accepted Samples", "True Model"]):
    x_min, x_max, y_min, y_max = [-10.0, 10.0, -10.0, 10.0]
    # plot_x, plot_y = np.linspace(x_min, x_max, n_plot), np.linspace(y_min, y_max, n_plot)
    # plot_x, plot_y = np.meshgrid(plot_x, plot_y)

    # grid_data = torch.tensor(list(zip(plot_x.reshape(-1), plot_y.reshape(-1))))
    # log_probs = model.log_prob(grid_data).detach().numpy()

    plt.scatter(samples_1[:, 0], samples_1[:, 1],
                label="True Model", s=4, color="#dd5182", alpha=0.5)

    plt.scatter(samples_2[:, 0], samples_2[:, 1],
                label="Learned Model", s=4, color="#ffa600", alpha=0.5)

    # c2 = plt.contour(plot_x, plot_y,
    #                  log_probs.reshape(n_plot, n_plot),
    #                  levels=50, linestyles="solid", cmap="viridis")

    # plt.colorbar(c2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("MCMC Samples vs Model Contour")
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    # plt.show()



def plot_hist(samples):
    if samples.size(1) == 1:
        return plot_hist_1D(samples)
    return plot_hist_2D(samples)

def plot_hist_1D(samples):
    ax = sns.distplot(samples, color = '#003f5c')
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Model")
    # plt.show()

def plot_hist_2D(samples):
    g = sns.jointplot(x=samples[:, 0], y=samples[:, 1], label="Samples",
                      s=4, color="#dd5182")
    g.set_axis_labels('X', 'Y')
    # plt.show()


def plot_mcmc(samples):
    if samples.size(1) == 1:
        return plot_mcmc_1D(samples)
    return plot_mcmc_2D(samples)

def plot_mcmc_1D(samples):
    x = np.linspace(0, len(samples), len(samples))
    y = samples.squeeze(1)
    sns.lineplot(x, y)
    plt.xlabel("Sample #")
    plt.ylabel("X_t Value")
    plt.title("MCMC Accepted Samples")
    # plt.plot()

def plot_mcmc_2D(samples):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    x = np.linspace(0, len(samples), len(samples))
    y_0 = samples[:, 0]
    y_1 = samples[:, 1]
    ax = sns.lineplot(x, y_0, ax=ax1)
    ax.set_title("MCMC Accepted Samples")
    ax.set_ylabel("X")
    ax = sns.lineplot(x, y_1, ax=ax2)
    ax.set_ylabel("Y")
    ax.set_xlabel("Sample #")
    # plt.plot()


def plot_loss_function_mu(loss, q_ref=Normal, p_model=Normal(), iterations=10):
    loss_values = []
    mus = np.linspace(-15, 15, 100)

    for mu in mus:
        q_model = q_ref(mu)

        values = []
        for _ in range(iterations):
            values.append(loss(p_model, q_model, 64).item())
        values = np.array(values)
        loss_values.append((values.mean(), values.std()))
    #     plt.plot(values)
    plt.plot(mus, [m for (m, _) in loss_values])
    plt.plot(mus, [m+s for (m, s) in loss_values])
    plt.plot(mus, [m-s for (m, s) in loss_values])
    plt.xlabel(r"Q's $\mu$ Point")
    plt.ylabel("Loss")
    plt.title("Learning Landscape")

def plot_loss_function_std(loss, q_ref=Normal, p_model=Normal(), iterations=10):
    loss_values = []
    mus = np.linspace(1e-4, 30, 100)

    for mu in mus:
        q_model = q_ref(0., mu)

        values = []
        for _ in range(iterations):
            values.append(loss(p_model, q_model, 64).item())
        values = np.array(values)
        loss_values.append((values.mean(), values.std()))
    #     plt.plot(values)
    plt.plot(mus, [m for (m, _) in loss_values])
    plt.plot(mus, [m+s for (m, s) in loss_values])
    plt.plot(mus, [m-s for (m, s) in loss_values])
    plt.xlabel(r"Q's $\sigma$ Point")
    plt.ylabel("Loss")
    plt.title("Learning Landscape")


def plot_loss_function(loss, q_ref=Normal, p_model=Normal(), n_plot=100, iterations=10):
    xlist = np.linspace(-15., 15.0, n_plot)
    ylist = np.linspace(1e-1, 50.0, n_plot)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros((len(ylist), len(xlist)))
    for _ in range(iterations):
        Z += np.array([[loss(p_model, q_ref(x, y), 128).item() for x in xlist] for y in ylist])
    Z /= iterations
    Z = np.log1p(Z-Z.min())
    cp = plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 100), cmap='RdGy')
    plt.title('Log Loss')
    plt.colorbar();
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\sigma$')


# EOF
