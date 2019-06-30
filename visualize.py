import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')

def plot_stats(stats, goals=None):
    palette = plt.get_cmap('Set1')
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
    plt.show()

def plot_models(p_model, q_model):
    if p_model.n_dims == 1:
        return plot_models_1D(p_model, q_model)
    return plot_models_2D(p_model, q_model)

def plot_models_1D(p_model, q_model):
    q_samples = q_model.sample(10000).detach().numpy()
    p_samples = p_model.sample(10000).detach().numpy()
    ax = sns.distplot(q_samples, color = '#003f5c', label="Learned Model")
    ax = sns.distplot(p_samples, color = '#ffa600', label="True Model")
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Models")
    plt.legend()
    plt.show()

# def get_dimensions(q_samples, p_samples):
#     x_min = min(min(q_samples[:, 0]), min(p_samples[:, 0]))
#     x_max = max(max(q_samples[:, 0]), max(p_samples[:, 0]))
#     y_min = min(min(q_samples[:, 1]), min(p_samples[:, 1]))
#     y_max = max(max(q_samples[:, 1]), max(p_samples[:, 1]))
#     return x_min-1, x_max+1, y_min-1, y_max+1

def plot_models_2D(p_model, q_model, n_plot=500):
    q_samples = q_model.sample(10000).detach().numpy()
    p_samples = p_model.sample(10000).detach().numpy()

    x_min, x_max, y_min, y_max = [-10.0, 10.0, -10.0, 10.0] #get_dimensions(q_samples, p_samples)
    plot_x, plot_y = np.linspace(x_min, x_max, n_plot), np.linspace(y_min, y_max, n_plot)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)

    grid_data = torch.tensor(list(zip(plot_x.reshape(-1), plot_y.reshape(-1))))
    log_probs = q_model.log_prob(grid_data).detach().numpy()

    plt.scatter(p_samples[:, 0], p_samples[:, 1],
                label="True Model", s=4, color="#dd5182")

    c2 = plt.contour(plot_x, plot_y,
                     log_probs.reshape(n_plot, n_plot),
                     levels=50, linestyles="solid", cmap="viridis")

    plt.colorbar(c2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Model Plot")
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.show()

def plot_model(model):
    if model.n_dims == 1:
        return plot_model_1D(model)
    return plot_model_2D(model)

def plot_model_1D(model):
    samples = model.sample(10000).detach().numpy()
    ax = sns.distplot(samples, color = '#003f5c', label="Model")
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Model")
    plt.show()

def plot_model_2D(model):
    plot_models_2D(model, model)

def plot_hists(samples_1, samples_2, labels=["Accepted Samples", "True Model"]):
    ax = sns.distplot(samples_1, color = '#003f5c', label=labels[0])
    ax = sns.distplot(samples_2, color = '#ffa600', label=labels[1])
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Model")
    plt.legend()
    plt.show()

def plot_hist(samples):
    ax = sns.distplot(samples, color = '#003f5c')
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Model")
    plt.show()


def plot_mcmc(samples):
    x = np.linspace(0, len(samples), len(samples))
    y = samples.squeeze(1).squeeze(1)
    sns.lineplot(x, y)
    plt.xlabel("Sample #")
    plt.ylabel("X_t Value")
    plt.title("MCMC Acceptance Samples")
    plt.plot()



# EOF
