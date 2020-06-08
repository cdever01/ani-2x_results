import pandas as pan
import numpy as np
import geom
import os

conf_data = pan.read_csv('conf_energies.csv', sep=',')

names = []
for structure in conf_data['Structure']:
    if structure[:-3] not in names:
        names.append(structure[:-3])

dft_rmsd = []
ani_rmsd = []
pm6_rmsd = []
mmff_rmsd = []

dft_bond_error = []
dft_angle_error = []
dft_dihedral_error = []

ani_bond_error = []
ani_angle_error = []
ani_dihedral_error = []

pm6_bond_error = []
pm6_angle_error = []
pm6_dihedral_error = []

mmff_bond_error = []
mmff_angle_error = []
mmff_dihedral_error = []

for name in names:
    for i in range(len(conf_data['Structure'])):
        if name in conf_data['Structure'][i]:
            mpf = '../mols/MP2/' + conf_data['Structure'][i] + '.mol'
            df = '../mols/wb97x/' + conf_data['Structure'][i] + '.mol'
            af = '../mols/ANI-2x/' + conf_data['Structure'][i] + '.mol'
            pf = '../mols/PM6/' + conf_data['Structure'][i] + '.mol'
            mf = '../mols/MMFF/' + conf_data['Structure'][i] + '.mol'
            db = geom.comp(mpf, df, 'bond')
            ab = geom.comp(mpf, af, 'bond')
            pb = geom.comp(mpf, pf, 'bond')
            mb = geom.comp(mpf, mf, 'bond')

            da = geom.comp(mpf, df, 'ang')
            aa = geom.comp(mpf, af, 'ang')
            pa = geom.comp(mpf, pf, 'ang')
            ma = geom.comp(mpf, mf, 'ang')

            dd = geom.comp(mpf, df, 'dih')
            ad = geom.comp(mpf, af, 'dih')
            pd = geom.comp(mpf, pf, 'dih')
            md = geom.comp(mpf, mf, 'dih')

            dft_rmsd.append(geom.get_rmsd(mpf, df))
            ani_rmsd.append(geom.get_rmsd(mpf, af))
            pm6_rmsd.append(geom.get_rmsd(mpf, pf))
            mmff_rmsd.append(geom.get_rmsd(mpf, mf))

            for b in range(len(ab)):
                dft_bond_error.append(db[b])
                ani_bond_error.append(ab[b])
                pm6_bond_error.append(pb[b])
                mmff_bond_error.append(mb[b])
            for a in range(len(aa)):
                dft_angle_error.append(da[a])
                ani_angle_error.append(aa[a])
                pm6_angle_error.append(pa[a])
                mmff_angle_error.append(ma[a])
            for d in range(len(ad)):
                dft_dihedral_error.append(dd[d])
                ani_dihedral_error.append(ad[d])
                pm6_dihedral_error.append(pd[d])
                mmff_dihedral_error.append(md[d])

dft_rmsd = np.array(dft_rmsd)
ani_rmsd = np.array(ani_rmsd)
pm6_rmsd = np.array(pm6_rmsd)
mmff_rmsd = np.array(mmff_rmsd)

dft_bond_error = np.array(dft_bond_error)
ani_bond_error = np.array(ani_bond_error)
pm6_bond_error = np.array(pm6_bond_error)
mmff_bond_error = np.array(mmff_bond_error)

dft_angle_error = np.array(dft_angle_error)
ani_angle_error = np.array(ani_angle_error)
pm6_angle_error = np.array(pm6_angle_error)
mmff_angle_error = np.array(mmff_angle_error)

dft_dihedral_error = np.array(dft_dihedral_error)
ani_dihedral_error = np.array(ani_dihedral_error)
pm6_dihedral_error = np.array(pm6_dihedral_error)
mmff_dihedral_error = np.array(mmff_dihedral_error)

dft_bond_mae = np.mean(dft_bond_error)
ani_bond_mae = np.mean(ani_bond_error)
pm6_bond_mae = np.mean(pm6_bond_error)
mmff_bond_mae = np.mean(mmff_bond_error)

dft_angle_mae = np.mean(dft_angle_error)
ani_angle_mae = np.mean(ani_angle_error)
pm6_angle_mae = np.mean(pm6_angle_error)
mmff_angle_mae = np.mean(mmff_angle_error)

dft_dihedral_mae = np.mean(dft_dihedral_error)
ani_dihedral_mae = np.mean(ani_dihedral_error)
pm6_dihedral_mae = np.mean(pm6_dihedral_error)
mmff_dihedral_mae = np.mean(mmff_dihedral_error)

Method = ['DFT', 'ANI', 'PM6', 'MMFF']
Bondsm = [dft_bond_mae, ani_bond_mae, pm6_bond_mae, mmff_bond_mae]
Angsm = [dft_angle_mae, ani_angle_mae, pm6_angle_mae, mmff_angle_mae]
Dihsm = [
    dft_dihedral_mae, ani_dihedral_mae, pm6_dihedral_mae, mmff_dihedral_mae
]
RMSDs = [
    np.mean(dft_rmsd),
    np.mean(ani_rmsd),
    np.mean(pm6_rmsd),
    np.mean(mmff_rmsd)
]
gdata = {}
gdata['Bonds MAE'] = Bondsm
gdata['Angles MAE'] = Angsm
gdata['Dihedrals MAE'] = Dihsm
gdata['RMSD'] = RMSDs

gd = pan.DataFrame(gdata, index=Method)
gd = gd.round(4)
print(gd)
