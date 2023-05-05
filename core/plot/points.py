import matplotlib.pyplot as plt
from matplotlib import ticker


def plot2d(
        ax,
        y,
        x,
        target_idx_colors=None,
        s=0.1,
        cmap='Accent',
        alpha=0.5,
        h_pad=None,
        w_pad=None,
        nth_dim=(0, 1),
        target_names=None,
        t: str = 'Graph'
):
    scatter = ax.scatter(
        x=x,
        y=y,
        c=target_idx_colors,
        s=s,
        cmap=cmap,
        alpha=alpha
    )

    handles, labels = scatter.legend_elements()
    labels = target_names
    ax.legend(handles, labels, loc="upper left", title="Topic")

    ax.set_xlabel(f'z{nth_dim[0]}')
    ax.set_ylabel(f'z{nth_dim[1]}')
    ax.set_title(t)
    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)
    plt.show()


def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, alpha=0.5)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)


def plot_2d(points, points_color, cpns, title):
    fig, ax = plt.subplots(
        figsize=(8, 7),
        facecolor="white",
        constrained_layout=True
    )
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color, cpns, title)
    plt.show()


def add_2d_scatter(ax, points, points_color, cpns, title=None):
    x, y = points[:, cpns[0]], points[:, cpns[1]]
    ax.scatter(
        x,
        y,
        c=points_color,
        # s=50,
        cmap='Accent',
        alpha=0.5
    )

    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # ax.set_xlabel(f'z{dim1} ')
    # ax.set_ylabel(f'z{dim2}')

    # handles, labels = scatter.legend_elements()
    # labels = topics
    # ax.legend(handles, labels, loc="upper left", title="Topic")

    # ax.set_xlabel(f'{dim1}th component')
    # ax.set_ylabel(f'{dim2}th component')

#%%
