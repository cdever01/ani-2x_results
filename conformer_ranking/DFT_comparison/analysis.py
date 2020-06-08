from itertools import combinations
import pandas as pd
import numpy as np
import os
import re
import torch
import torchani
from torchani.units import HARTREE_TO_KCALMOL
import scipy
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_xyz(file):
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


def all_diffs(lists):
    diffs = []
    for i in range(len(lists)):
        d = []
        for a, b in combinations(lists[i], 2):
            d.append(a - b)
        diffs.append(np.array(d))
    return diffs


def calculaterootmeansqrerror(data1, data2, axis=0):
    data = np.power(data1 - data2, 2)
    return np.sqrt(np.mean(data, axis=axis))


def calculatemeanabserror(data1, data2, axis=0):
    data = np.abs(data1 - data2)
    return np.mean(data, axis=axis)


model = ANI2x().to(device)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts(
    ['H', 'C', 'N', 'O', 'S', 'F', 'Cl'])

conf_data = pd.read_csv('conf_energies.csv', sep=',')

ANI_at_ANI_energies = []
ANI_at_DFT_energies = []

for structure in conf_data['Structure']:
    xyz, typ, Na, ct = read_xyz('../xyzs/ANI-2x/' + structure + '.xyz')
    coordinates = torch.tensor(xyz, requires_grad=True, device=device)
    species = species_to_tensor(typ[0]).unsqueeze(0).to(device)
    _, energy = model((species, coordinates))
    ANI_at_ANI_energies.append(energy.item() * HARTREE_TO_KCALMOL)

    xyz, typ, Na, ct = read_xyz('../xyzs/wb97x/' + structure + '.xyz')
    coordinates = torch.tensor(xyz, requires_grad=True, device=device)
    species = species_to_tensor(typ[0]).unsqueeze(0).to(device)
    _, energy = model((species, coordinates))
    ANI_at_DFT_energies.append(energy.item() * HARTREE_TO_KCALMOL)

ANI_at_ANI_energies = np.array(ANI_at_ANI_energies)
ANI_at_DFT_energies = np.array(ANI_at_DFT_energies)


def r2_and_sp(x, ys):
    sps = []
    rss = []
    for y in range(len(ys)):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            x, ys[y])
        sp, p = scipy.stats.spearmanr(x, ys[y])
        rs = r_value**2
        rss.append(rs)
        sps.append(sp)
    return rss, sps


names = []
for structure in conf_data['Structure']:
    if structure[:-3] not in names:
        names.append(structure[:-3])

ANI_at_ANI_dE = []
ANI_at_DFT_dE = []

DFT_at_DFT_dE = []
DFT_at_ANI_dE = []

ANI_at_ANI_sp = []
ANI_at_ANI_r2 = []

ANI_at_DFT_sp = []
ANI_at_DFT_r2 = []

ANI_DFT_sp = []
ANI_DFT_r2 = []

ANI_at_DFT_sp = []
ANI_at_DFT_r2 = []

DFT_at_ANI_sp = []
DFT_at_ANI_r2 = []

for name in names:
    ANI_at_ANI_conf = []
    ANI_at_DFT_conf = []
    DFT_at_ANI_conf = []
    DFT_at_DFT_conf = []
    for i in range(len(conf_data['Structure'])):
        if name in conf_data['Structure'][i]:
            ANI_at_ANI_conf.append(ANI_at_ANI_energies[i])
            ANI_at_DFT_conf.append(ANI_at_DFT_energies[i])
            DFT_at_ANI_conf.append(conf_data['DFT_at_ANI'][i])
            DFT_at_DFT_conf.append(conf_data['DFT_at_DFT'][i])
    a_d_rss, a_d_sps = r2_and_sp(DFT_at_DFT_conf, [ANI_at_ANI_conf])
    a_at_d_rss, a_at_d_sps = r2_and_sp(ANI_at_DFT_conf, [DFT_at_DFT_conf])
    d_at_a_rss, d_at_a_sps = r2_and_sp(DFT_at_ANI_conf, [ANI_at_ANI_conf])

    ANI_DFT_sp.append(a_d_sps[0])
    ANI_DFT_r2.append(a_d_rss[0])

    ANI_at_DFT_sp.append(a_at_d_sps[0])
    ANI_at_DFT_r2.append(a_at_d_rss[0])

    DFT_at_ANI_sp.append(d_at_a_sps[0])
    DFT_at_ANI_r2.append(d_at_a_rss[0])

    di = all_diffs(
        [ANI_at_ANI_conf, ANI_at_DFT_conf, DFT_at_DFT_conf, DFT_at_ANI_conf])
    for ANI_at_ANI, ANI_at_DFT, DFT_at_DFT, DFT_at_ANI in zip(
            di[0], di[1], di[2], di[3]):
        ANI_at_ANI_dE.append(ANI_at_ANI)
        ANI_at_DFT_dE.append(ANI_at_DFT)
        DFT_at_DFT_dE.append(DFT_at_DFT)
        DFT_at_ANI_dE.append(DFT_at_ANI)

ANI_at_ANI_dE = np.array(ANI_at_ANI_dE)
DFT_at_DFT_dE = np.array(DFT_at_DFT_dE)
ANI_at_DFT_dE = np.array(ANI_at_DFT_dE)
DFT_at_ANI_dE = np.array(DFT_at_ANI_dE)

ANI_DFT_sp = np.array(ANI_DFT_sp)
ANI_DFT_r2 = np.array(ANI_DFT_r2)

ANI_at_DFT_sp = np.array(ANI_at_DFT_sp)
ANI_at_DFT_r2 = np.array(ANI_at_DFT_r2)

DFT_at_ANI_sp = np.array(DFT_at_ANI_sp)
DFT_at_ANI_r2 = np.array(DFT_at_ANI_r2)

ANI_at_DFT_mae = calculatemeanabserror(ANI_at_DFT_dE, DFT_at_DFT_dE)
ANI_at_DFT_rmse = calculaterootmeansqrerror(ANI_at_DFT_dE, DFT_at_DFT_dE)

DFT_at_ANI_mae = calculatemeanabserror(DFT_at_ANI_dE, ANI_at_ANI_dE)
DFT_at_ANI_rmse = calculaterootmeansqrerror(DFT_at_ANI_dE, ANI_at_ANI_dE)

ANI_DFT_mae = calculatemeanabserror(ANI_at_ANI_dE, DFT_at_DFT_dE)
ANI_DFT_rmse = calculaterootmeansqrerror(ANI_at_ANI_dE, DFT_at_DFT_dE)

print('ANI at DFT MAE:', ANI_at_DFT_mae)
print('ANI at DFT RMSE:', ANI_at_DFT_rmse)
print('ANI at DFT mean spearman:', np.mean(ANI_at_DFT_sp))
print('ANI at DFT mean r squared:', np.mean(ANI_at_DFT_r2))

print('DFT at ANI MAE:', DFT_at_ANI_mae)
print('DFT at ANI RMSE:', DFT_at_ANI_rmse)
print('DFT at ANI mean spearman:', np.mean(DFT_at_ANI_sp))
print('DFT at ANI mean r squared:', np.mean(DFT_at_ANI_r2))

print('ANI_DFT MAE:', ANI_DFT_mae)
print('ANI_DFT RMSE:', ANI_DFT_rmse)
print('ANI_DFT mean spearman:', np.mean(ANI_DFT_sp))
print('ANI_DFT mean r squared:', np.mean(ANI_DFT_r2))
