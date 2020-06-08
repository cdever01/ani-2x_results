import pandas as pd
import numpy as np
import os
import re
import torch
import torchani
from torchani.units import HARTREE_TO_KCALMOL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_xyz(file):
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


model = ANI2x().to(device)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts(
    ['H', 'C', 'N', 'O', 'S', 'F', 'Cl'])

df = pd.read_csv('genetech_benchmark_data.csv')

ani_energies = []
cc_energies = []
dft_energies = []
opls_energies = []

for i in range(62):
    ani_energies_mol = []
    cc_energies_mol = []
    dft_energies_mol = []
    opls_energies_mol = []
    for j in range(36):
        cc_energies_mol.append(df['CC'][i * 36 + j])
        dft_energies_mol.append(df['WB97X'][i * 36 + j])
        opls_energies_mol.append(df['OPLS'][i * 36 + j])
        xyz, typ, Na, ct = read_xyz('xyzs/ANI-2x/' +
                                    df['Molecule'][i * 36 + j] + '.xyz')
        coordinates = torch.tensor(xyz, requires_grad=True, device=device)
        species = species_to_tensor(typ[0]).unsqueeze(0).to(device)
        _, energy = model((species, coordinates))
        ani_energies_mol.append(energy.item() * HARTREE_TO_KCALMOL)
    ani_energies_mol = np.array(ani_energies_mol)
    cc_energeis_mol = np.array(cc_energies_mol)
    dft_energies_mol = np.array(dft_energies_mol)
    opls_energies_mol = np.array(opls_energies_mol)

    ani_energies_mol -= min(ani_energies_mol)
    cc_energies_mol -= min(cc_energies_mol)
    dft_energies_mol -= min(dft_energies_mol)
    opls_energies_mol -= min(opls_energies_mol)

    ani_energies.extend(ani_energies_mol)
    cc_energies.extend(cc_energies_mol)
    dft_energies.extend(dft_energies_mol)
    opls_energies.extend(opls_energies_mol)

ani_energies = np.array(ani_energies)
cc_energies = np.array(cc_energies) * HARTREE_TO_KCALMOL
dft_energies = np.array(dft_energies)
opls_energies = np.array(opls_energies)

ANI_mae = calculatemeanabserror(ani_energies, cc_energies)
ANI_rmse = calculaterootmeansqrerror(ani_energies, cc_energies)

DFT_mae = calculatemeanabserror(dft_energies, cc_energies)
DFT_rmse = calculaterootmeansqrerror(dft_energies, cc_energies)

OPLS_mae = calculatemeanabserror(opls_energies, cc_energies)
OPLS_rmse = calculaterootmeansqrerror(opls_energies, cc_energies)

print('ANI MAE:', ANI_mae)
print('ANI RMSE:', ANI_rmse)
print('DFT MAE:', DFT_mae)
print('DFT RMSE:', DFT_rmse)
print('OPLS MAE:', OPLS_mae)
print('OPLS RMSE:', OPLS_rmse)
