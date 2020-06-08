from ase_interface import ensemblemolecule
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from data_reader import readncdatall
import torch
import torchani
from torchani.units import HARTREE_TO_KCALMOL
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_xyz(file):  #XYZ file reader for RXN
    import numpy as np
    xyz = []
    typ = []
    Na = []
    ct = []
    fd = open(file, 'r').read()
    rb = re.compile('(\d+?)\n(.*?)\n((?:[A-Z][a-z]?.+?(?:\n|$))+)')
    ra = re.compile(
        '([A-Z][a-z]?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s*?(?:\n|$)'
    )
    s = rb.findall(fd)
    Nc = len(s)
    if Nc == 0:
        raise ValueError('No coordinates found in file. Check formatting of ' +
                         file + '.')
    for i in s:
        X = []
        T = []
        ct.append(i[1])
        c = ra.findall(i[2])
        Na.append(len(c))
        for j in c:
            T.append(j[0])
            X.append(j[1])
            X.append(j[2])
            X.append(j[3])
        X = np.array(X, dtype=np.float32)
        X = X.reshape(len(T), 3)
        xyz.append(X)
        typ.append(T)

    return xyz, typ, Na, ct


model = torchani.models.ANI2x(periodic_table_index=False).to(device)


def calculaterootmeansqrerror(data1, data2, axis=0):
    data = np.power(data1 - data2, 2)
    return np.sqrt(np.mean(data, axis=axis))


def calculatemeanabserror(data1, data2, axis=0):
    data = np.abs(data1 - data2)
    return np.mean(data, axis=axis)


ha_to_kcal = 627.5094738898777

model = ANI2x().to(device)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts(
    ['H', 'C', 'N', 'O', 'S', 'F', 'Cl'])

data = read_xyz('final_1ps_traj.xyz')
X, S, Na, C = data
s = S[0]
species = species_to_tensor(s).unsqueeze(0).to(device)
dft_data = readncdatall('final_1ps_traj.dat')

Edft = dft_data['energies']
Fdft = dft_data['forces']

Fdft = np.linalg.norm(Fdft.reshape((-1, 3)), axis=1).reshape((-1, Na[0]))

E = []
F = []
for x in X:
    coordinates = torch.tensor(
        x, requires_grad=True, device=device).unsqueeze(0)
    _, energy = model((species, coordinates))
    derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
    force = -derivative
    force = force.cpu().numpy()
    force = force.squeeze()
    E.append(energy.item())
    F.append(np.linalg.norm(force, axis=1))
F = np.array(F)

F = np.stack(F)

Era = np.array(E) - np.array(E).mean()
Edf = np.array(Edft) - np.array(Edft).mean()

f, ax = plt.subplots(
    2, sharex=True, figsize=(10, 5), dpi=300, facecolor='w', edgecolor='k')
t = 25 * np.arange(1, Era.size + 1) * 0.4 / 1000

props = dict(boxstyle='round', facecolor='white', alpha=0.5)

Emae = HARTREE_TO_KCALMOL * calculatemeanabserror(Era, Edf)
Erms = HARTREE_TO_KCALMOL * calculaterootmeansqrerror(Era, Edf)
errorstr = "MAE:   " + "{0:.2f}".format(Emae) + "\nRMSE: " + "{0:.2f}".format(
    Erms)

ax[0].plot(
    t,
    HARTREE_TO_KCALMOL * Edf,
    '-',
    color='black',
    linewidth=1,
    label=r'$E_{DFT}$')
ax[0].plot(
    t,
    HARTREE_TO_KCALMOL * Era,
    '-',
    color='green',
    linewidth=1,
    label=r'$E_{ANI}$')
ax[0].plot(
    t,
    HARTREE_TO_KCALMOL * np.abs(Era - Edf),
    '-',
    color='red',
    linewidth=1,
    label=r'$\left|E_{ANI}-E_{DFT}\right|$')
ax[0].set_xlim([0, t[-1] + 0.001 + 6.0])
ax[0].axvline(25.0, color='black', linewidth=0.8)
ax[0].text(
    0.845,
    0.48,
    errorstr,
    transform=ax[0].transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=props)
ax[0].set_ylabel(r'$kcal \times mol^{-1}$', fontsize=12)
ax[0].legend(loc=1, fontsize=11, bbox_to_anchor=(1.0075, 1.02), frameon=False)
ax[0].set_title("Energy")

Fmae = HARTREE_TO_KCALMOL * calculatemeanabserror(F.flatten(), Fdft.flatten())
Frms = HARTREE_TO_KCALMOL * calculaterootmeansqrerror(F.flatten(),
                                                      Fdft.flatten())
errorstr = "MAE:   " + "{0:.2f}".format(Fmae) + "\nRMSE: " + "{0:.2f}".format(
    Frms)

ax[1].plot(
    t,
    HARTREE_TO_KCALMOL * (np.mean(np.abs(Fdft), axis=1)),
    '-',
    color='black',
    linewidth=1,
    label=r'$\left<\left|\left|F_{DFT}\right|\right|\right>$')
ax[1].plot(
    t,
    HARTREE_TO_KCALMOL * (np.mean(np.abs(F), axis=1)),
    '--',
    color='green',
    linewidth=1,
    label=r'$\left<\left|\left|F_{ANI}\right|\right|\right>$')
ax[1].plot(
    t,
    HARTREE_TO_KCALMOL * (np.mean(np.abs(F - Fdft), axis=1)),
    '-',
    color='red',
    linewidth=1,
    label=r'$\left<\left|\left|F_{ANI}-F_{DFT}\right|\right|\right>$')
ax[1].axvline(25.0, color='black', linewidth=0.8)
ax[1].text(
    0.845,
    0.29,
    errorstr,
    transform=ax[1].transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=props)
ax[1].set_xlim([0, t[-1] + 0.001 + 6.0])
ax[1].set_xlabel('Time (ps)', fontsize=12)
ax[1].set_ylabel(r'$kcal \times mol^{-1} \times \AA^{-1}$', fontsize=12)
ax[1].legend(loc=1, fontsize=10, bbox_to_anchor=(1.0075, 0.8), frameon=False)
ax[1].set_xticks([0., 5., 10., 15., 20, 25])
ax[1].set_title("Force Magnitude")

im = plt.imread(get_sample_data('/home/cdever01/GSK1107112A.png'))
newax = f.add_axes([0.705, 0.4, 0.2, 0.2], anchor='NE', zorder=100)
newax.imshow(im)
newax.text(
    0.49,
    0.83,
    "GSK1107112A",
    transform=newax.transAxes,
    fontsize=12,
    verticalalignment='center',
    horizontalalignment='center')
newax.set_xticks([])
newax.set_yticks([])

plt.show()
plt.savefig('trajectory.png')

print(t[-1])
print(Emae, Erms)
print(
    HARTREE_TO_KCALMOL * calculatemeanabserror(F.flatten(), Fdft.flatten()),
    HARTREE_TO_KCALMOL * calculaterootmeansqrerror(F.flatten(),
                                                   Fdft.flatten()))
