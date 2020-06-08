import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.image as mpimg


def plot_2d(X, Y, Z, xax='', yax='', save='2_D', label='Delta E (kcal/mol)'):

    vmin = np.min(np.array([Z]))
    vmax = np.max(np.array([Z]))

    X = np.array(X)
    Y = np.array(Y)
    for i in range(len(Z)):
        Z[i] = np.reshape(Z[i], (len(X), len(Y)))
        Z[i] = Z[i][::-1, :]

    fig, ax = plt.subplots(1, len(Z), figsize=(9, 9))

    ax[0].set_ylabel(yax, fontsize=30, labelpad=-20)
    for a in range(len(ax)):
        im = ax[a].imshow(
            Z[a].T,
            interpolation='gaussian',
            extent=[min(X), max(X), min(Y), max(Y)],
            cmap=mpl.cm.jet,
            alpha=1.0,
            vmax=vmax)
        levels = np.arange(0.0, Z[a].max(), 6)
        CS = ax[a].contour(
            Z[a][:, ::-1].T,
            levels,
            extent=(min(X), max(X), min(Y), max(Y)),
            colors='black')
        ax[a].set_xlabel(xax, fontsize=30)
        ax[a].tick_params(labelsize=20)
        ax[a].yaxis.set_ticks(np.arange(-160, 161, 80))
        ax[a].xaxis.set_ticks(np.arange(-160, 161, 80))

    fig.tight_layout()

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.9, 0.3, 0.05, 0.4])
    cb = fig.colorbar(
        im, cax=cbar_ax, shrink=.5).set_label(
            label=label, size=30, weight='bold')
    axe = cbar_ax.tick_params(labelsize=30)

    fig.show()

    fig.savefig(save + '.png', bbox_inches='tight', transparent=True)
