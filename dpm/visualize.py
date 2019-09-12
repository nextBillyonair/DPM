import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib
import seaborn as sns
from dpm.distributions import Uniform, Normal, Data
from dpm.criterion import make_distance_matrix
from torch.nn.functional import softplus
import torch
plt.style.use('seaborn-darkgrid')

color_seq = { 0:"#003f5c", 1:"#444e86", 2:"#955196", 3:"#dd5182", 4:"#ff6e54", 5:"#ffa600"}

def plot_stats(stats, goals=None, axes=None, offset=0):
    if axes is None:
        fig, axes = plt.subplots(1, len(stats.data.keys()), figsize=(12, 6), squeeze=False)

    for index, value in enumerate(stats.data.keys()):
        axes[0, index].set_title(f"{value} curve")
        axes[0, index].set_xlabel('Epoch')
        axes[0, index].set_ylabel(value)
        axes[0, index].plot(np.arange(0.0, len(stats.data[value]), 1.0),
                         stats.data[value], color=color_seq[(offset + index) % len(color_seq)])
    if goals is not None:
        for i, goal in enumerate(goals):
            axes[0, i+1].axhline(goals[i], color="#ff6e54", linewidth=4)
    plt.tight_layout()


def plot_models(p_model, q_model, batch_size=10000, n_plot=500):
    if p_model.n_dims == 1:
        return plot_models_1D(p_model, q_model, batch_size)
    return plot_models_2D(p_model, q_model, batch_size, n_plot)

def plot_models_1D(p_model, q_model, batch_size=10000):
    p_samples = p_model.sample(batch_size).detach().numpy()
    q_samples = q_model.sample(batch_size).detach().numpy()
    ax = sns.distplot(q_samples, color = '#003f5c', label="Learned Model")
    ax = sns.distplot(p_samples, color = '#ffa600', label="True Model")
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Models")
    plt.legend()


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

def plot_contour(model, n_plot=500, rng=(-10, 10)):
    if model.n_dims == 2:
        return plot_contour_2d(model, n_plot, rng=rng)
    return plot_contour_1d(model, n_plot, rng=rng)

def plot_contour_1d(model, n_plot=500, rng=(-10, 10)):
    X = np.linspace(rng[0], rng[1], n_plot).reshape(-1, 1)
    probs = model.log_prob(torch.tensor(X).view(-1, 1).float()).exp().detach().numpy()
    probs = np.nan_to_num(probs)
    plt.plot(X, probs, label="Model Prob", color="#dd5182")
    plt.fill(X, probs, color="#dd5182", alpha=0.3)
    plt.xlabel("X")
    plt.ylabel("Prob")
    plt.title("P Samples vs Q Contour")
    plt.xlim(rng[0], rng[1]); plt.ylim(0, min(max(0.5, probs.max()+.05), 10.))


def plot_contour_2d(model, n_plot=500, rng=(-10, 10)):
    plot_x, plot_y = np.linspace(rng[0], rng[1], n_plot), np.linspace(rng[0], rng[1], n_plot)
    plot_x, plot_y = np.meshgrid(plot_x, plot_y)

    grid_data = torch.tensor(list(zip(plot_x.reshape(-1), plot_y.reshape(-1))))
    log_probs = model.log_prob(grid_data).detach().numpy()
    c2 = plt.contour(plot_x, plot_y,
                     log_probs.reshape(n_plot, n_plot),
                     levels=50, linestyles="solid", cmap="viridis")
    plt.colorbar(c2)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Model Contour")
    plt.xlim(rng[0], rng[1]); plt.ylim(rng[0], rng[1])


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

def plot_model_2D(model, batch_size=10000):
    plot_models_2D(model, model, batch_size)



def plot_hists(samples_1, samples_2, labels=["Accepted Samples", "True Model"], bins=50):
    if samples_1.size(1) == 1:
        return plot_hists_1D(samples_1.detach(), samples_2.detach(), labels, bins)
    return plot_hists_2D(samples_1.detach(), samples_2.detach(), labels)

def plot_hists_1D(samples_1, samples_2, labels=["Accepted Samples", "True Model"], bins=50):
    ax = sns.distplot(samples_1, color = '#003f5c', label=labels[0], bins=bins)
    ax = sns.distplot(samples_2, color = '#ffa600', label=labels[1], bins=bins)
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Model")
    plt.legend()

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



def plot_hist(samples):
    if samples.size(1) == 1:
        return plot_hist_1D(samples.detach())
    return plot_hist_2D(samples.detach())

def plot_hist_1D(samples):
    ax = sns.distplot(samples, color = '#003f5c')
    plt.xlabel("Sample")
    plt.ylabel("Density")
    plt.title("Distplot for Model")

def plot_hist_2D(samples):
    g = sns.jointplot(x=samples[:, 0], y=samples[:, 1], label="Samples",
                      s=4, color="#dd5182")
    g.set_axis_labels('X', 'Y')


def plot_mcmc(samples):
    if samples.size(1) == 1:
        return plot_mcmc_1D(samples)
    return plot_mcmc_2D(samples)

def plot_mcmc_1D(samples):
    x = np.linspace(0, len(samples), len(samples))
    y = samples.squeeze(1)
    sns.lineplot(x, y)
    plt.xlabel("Sample #")
    plt.ylabel(r"$X_t$ Value")
    plt.title("MCMC Accepted Samples")

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


# def plot_loss_function_mu(loss, q_ref=Normal, p_model=Normal(), iterations=10):
#     loss_values = []
#     mus = np.linspace(-15, 15, 100)
#
#     for mu in mus:
#         q_model = q_ref(mu)
#
#         values = []
#         for _ in range(iterations):
#             values.append(loss(p_model, q_model, 64).item())
#         values = np.array(values)
#         loss_values.append((values.mean(), values.std()))
#     #     plt.plot(values)
#     plt.plot(mus, [m for (m, _) in loss_values])
#     plt.plot(mus, [m+s for (m, s) in loss_values])
#     plt.plot(mus, [m-s for (m, s) in loss_values])
#     plt.xlabel(r"Q's $\mu$ Point")
#     plt.ylabel("Loss")
#     plt.title("Learning Landscape")
#
# def plot_loss_function_std(loss, q_ref=Normal, p_model=Normal(), iterations=10):
#     loss_values = []
#     mus = np.linspace(1e-4, 30, 100)
#
#     for mu in mus:
#         q_model = q_ref(0., mu)
#
#         values = []
#         for _ in range(iterations):
#             values.append(loss(p_model, q_model, 64).item())
#         values = np.array(values)
#         loss_values.append((values.mean(), values.std()))
#     #     plt.plot(values)
#     plt.plot(mus, [m for (m, _) in loss_values])
#     plt.plot(mus, [m+s for (m, s) in loss_values])
#     plt.plot(mus, [m-s for (m, s) in loss_values])
#     plt.xlabel(r"Q's $\sigma$ Point")
#     plt.ylabel("Loss")
#     plt.title("Learning Landscape")


def plot_loss_function(loss, q_ref=Normal, p_model=Normal(), n_plot=100, batch_size=64):
    xlist = np.linspace(-15., 15.0, n_plot)
    ylist = np.linspace(1e-1, 50.0, n_plot)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.array([[loss(p_model, q_ref(x, y), batch_size).item() for x in xlist] for y in ylist])
    Z = np.log1p(Z-Z.min())
    cp = plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 100), cmap='RdGy')
    plt.title('Log Loss')
    plt.colorbar();
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\sigma$')



def get_emd_colormap(vmin=0, vmax=10, cmap=cm.rainbow):
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    return cm.ScalarMappable(norm=cNorm, cmap=cmap)


def plot_emd_hist(hist, title=r"", ylim=0.25, cmap=cm.rainbow, colorMap=None):
    if colorMap is None:
        colorMap = get_emd_colormap(vmin=0, vmax=len(hist), cmap=cm.rainbow)

    for i in range(len(hist)):
    	plt.bar(i, hist[i], 1, color=colorMap.to_rgba(i), edgecolor="white", linewidth=1)

    plt.title(title, y=-0.2, x=0.5, fontsize=20)
    plt.axis('off')
    plt.ylim(0, ylim)


def plot_emd_gamma(gamma):
    D = make_distance_matrix(gamma.shape[0], gamma.shape[1])
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)

    ax[0].imshow(gamma, cmap=cm.gist_heat, interpolation='nearest')
    ax[0].axis('off')
    ax[0].set_title(r"$\mathbf{\Gamma}$", y=-0.13, fontsize=40)
    ax[1].imshow(D, cmap=cm.gist_heat, interpolation='nearest')
    ax[1].set_title(r"$\mathbf{D}$", y=-0.13, fontsize=40)
    ax[1].axis('off')


def plot_emd_partition(gamma, colorMap, titles=[r"", r""], ylim=0.25):
    fig, ax = plt.subplots(2, 1)
    fig.set_figheight(6)
    fig.set_figwidth(10)

    pr_len = gamma.shape[0]
    pt_len = gamma.shape[1]

    r = range(pr_len)
    current_bottom = np.zeros(pr_len)

    for i in range(pt_len).__reversed__():
    	ax[0].bar(r, gamma[r, i], 1, color=colorMap.to_rgba(r), bottom=current_bottom,
                edgecolor="white", linewidth=1)
    	current_bottom = current_bottom + gamma[r, i]

    ax[0].axis('off')
    ax[0].set_ylim(0, ylim)
    ax[0].set_title(titles[0], y=-0.25, x=0.5, fontsize=20)

    current_bottom = np.zeros(pt_len)
    r = range(pt_len)

    for i in range(pr_len):
    	ax[1].bar(r, gamma[i, r], 1, color=colorMap.to_rgba(i), bottom=current_bottom,
                edgecolor="white", linewidth=1)
    	current_bottom = current_bottom + gamma[i, r]

    ax[1].axis('off')
    ax[1].set_ylim(0, ylim)
    ax[1].set_title(titles[1], y=-0.25, x=0.5, fontsize=20)





# EOF
