import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdt
import hdnntools as  hdn
import numpy as np

import pybel
import openbabel as ob

import os



def enumerateAngles(f):                             #openbabel
	f_name, f_ext = os.path.splitext(f)
	obconversion = ob.OBConversion()
	obconversion.SetInFormat(f_ext)
	obmol = ob.OBMol()
	this=obconversion.ReadFile( obmol, f )
	all_angles=[i for i in ob.OBMolAngleIter(obmol)]
	return all_angles



def enumerateDihs(f):                             #openbabel
        f_name, f_ext = os.path.splitext(f)
        obconversion = ob.OBConversion()
        obconversion.SetInFormat(f_ext)
        obmol = ob.OBMol()
        this=obconversion.ReadFile( obmol, f )
        all_dihs=[i for i in ob.OBMolTorsionIter(obmol)]
        return all_dihs


def enumerateTorsions(mol):                          #rdkit
   torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
   torsionQuery = Chem.MolFromSmarts(torsionSmarts)
   matches = mol.GetSubstructMatches(torsionQuery)
   torsionList = []
   for match in matches:
     idx2 = match[0]
     idx3 = match[1]
     bond = mol.GetBondBetweenAtoms(idx2, idx3)
     jAtom = mol.GetAtomWithIdx(idx2)
     kAtom = mol.GetAtomWithIdx(idx3)
     if (((jAtom.GetHybridization() != Chem.HybridizationType.SP2)
       and (jAtom.GetHybridization() != Chem.HybridizationType.SP3))
       or ((kAtom.GetHybridization() != Chem.HybridizationType.SP2)
       and (kAtom.GetHybridization() != Chem.HybridizationType.SP3))):
       continue
     for b1 in jAtom.GetBonds():
       if (b1.GetIdx() == bond.GetIdx()):
         continue
       idx1 = b1.GetOtherAtomIdx(idx2)
       for b2 in kAtom.GetBonds():
         if ((b2.GetIdx() == bond.GetIdx())
           or (b2.GetIdx() == b1.GetIdx())):
           continue
         idx4 = b2.GetOtherAtomIdx(idx3)
         # skip 3-membered rings
         if (idx4 == idx1):
           continue
         torsionList.append((idx1, idx2, idx3, idx4))
   return torsionList



def enumerateBonds(f):                                       #rdkit
	mol=Chem.MolFromMolFile(f, removeHs=False, sanitize=False)
	c=mol.GetConformer()
	atid=[]
	for i in range(len(mol.GetBonds())):
		s=mol.GetBondWithIdx(i).GetBeginAtomIdx()
		e=mol.GetBondWithIdx(i).GetEndAtomIdx()
		atid.append((s,e))
	return atid



def get_bonds(f, atid=None):                                   #rdkit
	mol=Chem.MolFromMolFile(f, removeHs=False, sanitize=False)
	c=mol.GetConformer()
	bonds=[]
	if atid==None:
		atid=enumerateBonds(f)
	for i in range(len(atid)):
		bonds.append(rdt.GetBondLength(c, atid[i][0], atid[i][1]))
	
	return np.array(bonds), atid



def get_angs(f, atid=None):                                   #rdkit 
	mol=Chem.MolFromMolFile(f, removeHs=False, sanitize=False)
	c=mol.GetConformer()
	if atid==None:
		atid=enumerateAngles(f)
	angs=[]
	for i in range(len(atid)):
		m=mol.GetAtomWithIdx(atid[i][1])
		if m.GetHybridization() != Chem.HybridizationType.SP:
			angs.append(rdt.GetAngleDeg(c, atid[i][0], atid[i][1], atid[i][2]))
	return np.array(angs), atid


def get_dihs(f, atid=None):                                     #rdkit
	mol=Chem.MolFromMolFile(f, removeHs=False, sanitize=False)
	c=mol.GetConformer()
	if atid==None:
		atid=enumerateDihs(f)
	dih=[]
	for i in range(len(atid)):
		bond = mol.GetBondBetweenAtoms(atid[i][1], atid[i][2])
		a2=mol.GetAtomWithIdx(atid[i][1])
		a3=mol.GetAtomWithIdx(atid[i][2])
		if a2.GetHybridization!=Chem.HybridizationType.SP and a3.GetHybridization!=Chem.HybridizationType.SP:
			dih.append(rdt.GetDihedralDeg(c, atid[i][0], atid[i][1], atid[i][2], atid[i][3]))
	return np.array(dih), atid




def get_rmsd(ref, tar, H=False, sym=True):                     #openbabel
	r_name, r_ext = os.path.splitext(ref)
	obconversion = ob.OBConversion()
	obconversion.SetInFormat(r_ext)
	obmol = ob.OBMol()
	this=obconversion.ReadFile(obmol, ref)
	t_name, t_ext = os.path.splitext(tar)
	obconversion2 = ob.OBConversion()
	obconversion2.SetInFormat(t_ext)
	obmol2 = ob.OBMol()
	that=obconversion.ReadFile( obmol2, tar)
	
	aligner = ob.OBAlign(H, sym)  # includeH, symmetry
	aligner.SetMethod(1)
	aligner.SetRefMol(obmol)
	aligner.SetTargetMol(obmol2)
	aligner.Align()
	rms = aligner.GetRMSD()
	return rms





def comp(f1, f2, prop):
	if prop=='bond':
		prop1, atid=get_bonds(f1)
		prop2, atid=get_bonds(f2, atid)
		delta=np.abs(prop1-prop2)
	elif prop=='ang':
		prop1, atid=get_angs(f1)
		prop2, atid=get_angs(f2, atid)
		delta=np.abs(prop1-prop2)%360
	elif prop=='dih':
		prop1, atid=get_dihs(f1)
		prop2, atid=get_dihs(f2, atid)
		delta=180.0-np.abs(np.abs(prop1-prop2)-180)
	rmsd=get_rmsd(f1, f2)
	return delta


