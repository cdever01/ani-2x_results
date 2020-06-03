import matplotlib as mpl
mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.image as mpimg

def plot_2d(X, Y, Z, xax='', yax='', save='2_D', titles=[], fs=30, cbx=[0.9, 0.3, 0.05, 0.4], pad=-20, size=(9,9), trans=True, cont=6, ticks=np.arange(-160,161,10), label='Delta E (kcal/mol)'):

    vmin = np.min(np.array([Z]))
    vmax = np.max(np.array([Z]))


    X=np.array(X)
    Y=np.array(Y)
    for i in range(len(Z)):
        Z[i]=np.reshape(Z[i], (len(X), len(Y)))
        Z[i]=Z[i][::-1,:]




    fig, ax = plt.subplots(1,len(Z), figsize=size)
    if len(Z)==1:
        ax=[ax]

    if titles==[]:
        for i in range(len(Z)):
            titles.append(str(i))

    ax[0].set_ylabel(yax, fontsize=fs, labelpad=pad)
    for a in range(len(ax)):
        im = ax[a].imshow(Z[a].T, interpolation='gaussian', extent=[min(X), max(X), min(Y), max(Y)], cmap=mpl.cm.jet, alpha=1.0, vmax=vmax)
        levels = np.arange(0.0, Z[a].max(), cont)
        CS=ax[a].contour(Z[a][:,::-1].T, levels, extent=(min(X), max(X), min(Y), max(Y)), colors='black')
        ax[a].set_title(titles[a], fontsize=40)
        ax[a].set_xlabel(xax, fontsize=fs)
        #ax[a].set_ylabel(yax, fontsize=50, labelpad=-30)
        ax[a].tick_params(labelsize=fs)
        ax[a].yaxis.set_ticks(ticks)
        ax[a].xaxis.set_ticks(ticks)

    fig.tight_layout()

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes(cbx)
    cb = fig.colorbar(im, cax=cbar_ax, shrink=.5).set_label(label=label,size=fs,weight='bold')
    axe = cbar_ax.tick_params(labelsize=fs)



    fig.show()

    fig.savefig(save+'.png',bbox_inches='tight', transparent=trans)
