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


def read_xyz(file):  #XYZ file reader for RXN
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

ANI_2x_energies = []

for structure in conf_data['Structure']:
    xyz, typ, Na, ct = read_xyz('../xyzs/ANI-2x/' + structure + '.xyz')
    coordinates = torch.tensor(xyz, requires_grad=True, device=device)
    species = species_to_tensor(typ[0]).unsqueeze(0).to(device)
    _, energy = model((species, coordinates))
    ANI_2x_energies.append(energy.item() * HARTREE_TO_KCALMOL)

ANI_2x_energies = np.array(ANI_2x_energies)


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

ANI_dE = []
DFT_dE = []
MP2_dE = []
PM6_dE = []
MMFF_dE = []

ANI_sp = []
DFT_sp = []
PM6_sp = []
MMFF_sp = []

ANI_r2 = []
DFT_r2 = []
PM6_r2 = []
MMFF_r2 = []

for name in names:
    ANI_conf = []
    DFT_conf = []
    MP2_conf = []
    PM6_conf = []
    MMFF_conf = []
    for i in range(len(conf_data['Structure'])):
        if name in conf_data['Structure'][i]:
            ANI_conf.append(ANI_2x_energies[i])
            DFT_conf.append(conf_data['wb97x'][i])
            MP2_conf.append(conf_data['MP2'][i])
            PM6_conf.append(conf_data['PM6'][i])
            MMFF_conf.append(conf_data['MMFF'][i])

    rss, sps = r2_and_sp(MP2_conf, [ANI_conf, DFT_conf, PM6_conf, MMFF_conf])

    ANI_sp.append(sps[0])
    ANI_r2.append(rss[0])
    DFT_sp.append(sps[1])
    DFT_r2.append(rss[1])
    PM6_sp.append(sps[2])
    PM6_r2.append(rss[2])
    MMFF_sp.append(sps[3])
    MMFF_r2.append(rss[3])

    di = all_diffs([ANI_conf, DFT_conf, MP2_conf, PM6_conf, MMFF_conf])
    for ANI, DFT, MP2, PM6, MMFF in zip(di[0], di[1], di[2], di[3], di[4]):
        ANI_dE.append(ANI)
        DFT_dE.append(DFT)
        MP2_dE.append(MP2)
        PM6_dE.append(PM6)
        MMFF_dE.append(MMFF)
ANI_dE = np.array(ANI_dE)
DFT_dE = np.array(DFT_dE)
MP2_dE = np.array(MP2_dE)
PM6_dE = np.array(PM6_dE)
MMFF_dE = np.array(MMFF_dE)

ANI_sp = np.array(ANI_sp)
ANI_r2 = np.array(ANI_r2)
DFT_sp = np.array(DFT_sp)
DFT_r2 = np.array(DFT_r2)
PM6_sp = np.array(PM6_sp)
PM6_r2 = np.array(PM6_r2)
MMFF_sp = np.array(MMFF_sp)
MMFF_r2 = np.array(MMFF_r2)

ANI_mae = calculatemeanabserror(ANI_dE, MP2_dE)
DFT_mae = calculatemeanabserror(DFT_dE, MP2_dE)
PM6_mae = calculatemeanabserror(PM6_dE, MP2_dE)
MMFF_mae = calculatemeanabserror(MMFF_dE, MP2_dE)

ANI_rmse = calculaterootmeansqrerror(ANI_dE, MP2_dE)
DFT_rmse = calculaterootmeansqrerror(DFT_dE, MP2_dE)
PM6_rmse = calculaterootmeansqrerror(PM6_dE, MP2_dE)
MMFF_rmse = calculaterootmeansqrerror(MMFF_dE, MP2_dE)

print('ANI MAE:', ANI_mae)
print('ANI RMSE:', ANI_rmse)
print('ANI mean spearman:', np.mean(ANI_sp))
print('ANI mean r squared:', np.mean(ANI_r2))
print('DFT MAE:', DFT_mae)
print('DFT RMSE:', DFT_rmse)
print('DFT mean spearman:', np.mean(DFT_sp))
print('DFT mean r squared:', np.mean(DFT_r2))
print('PM6 MAE:', PM6_mae)
print('PM6 RMSE:', PM6_rmse)
print('PM6 mean spearman:', np.mean(PM6_sp))
print('PM6 mean r squared:', np.mean(PM6_r2))
print('MMFF MAE:', MMFF_mae)
print('MMFF RMSE:', MMFF_rmse)
print('MMFF mean spearman:', np.mean(MMFF_sp))
print('MMFF mean r squared:', np.mean(MMFF_r2))
