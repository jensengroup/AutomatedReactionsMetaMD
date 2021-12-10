import sys
import os
import tarfile
import pandas as pd
import numpy as np
from rdkit import Chem


def canonicalize_smiles(structure_smiles):
    """
    remove all structural info an atom mapping information
    """
    mol = Chem.MolFromSmiles(structure_smiles, sanitize=False)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    canonical_smiles = Chem.MolToSmiles(mol)

    return canonical_smiles


def get_complex_energies(r_idx):
    """
    check the optimized xTB structures from either side of metadynamics
    calculation and extract optimized energies. 
    """
    global COMPLEX_ENERGIES
    os.chdir(r_idx)
    rxn_df = pd.read_csv(r_idx+'_1step.csv', index_col=0)
    for run in range(100):
        try:
            run_df = pd.read_pickle('run'+str(run)+'.pkl')
        except FileNotFoundError:
            continue
        for reaction in run_df.Reactions:
            if reaction in rxn_df.Reactions.to_list() or reaction.split('>>')[1]+'>>'+reaction.split('>>')[0] in rxn_df.Reactions.to_list():
                tar = tarfile.open('run'+str(run)+'_database.tar.gz', "r:gz")
                structure_df = tar.extractfile('run'+str(run)+'/structure_database/structure_database.csv')
                structure_df = pd.read_csv(structure_df, index_col=0)
                r_smiles = reaction.split('>>')[0]
                r_canonical = canonicalize_smiles(r_smiles)
                p_smiles = reaction.split('>>')[1]
                p_canonical = canonicalize_smiles(p_smiles)
                r_hash = structure_df.index[structure_df.smiles == r_smiles].to_list()[0]
                try:
                    n_reactant = len(COMPLEX_ENERGIES.loc[r_canonical])
                except KeyError:
                    n_reactant = 0
                try:
                    n_product = len(COMPLEX_ENERGIES.loc[p_canonical])
                except KeyError:
                    n_product = 0
                p_hash = structure_df.index[structure_df.smiles == p_smiles].to_list()[0]
                #print(structure_df)
                r_file_name = 'run'+str(run)+'/structure_database/xyz_files/'+str(r_hash)+'.xyz'
                r_file_name = tar.extractfile(r_file_name)
                r_file = r_file_name.read().decode('utf-8')
                r_lines = r_file.split('\n')
                #print(r_lines)
                p_file_name = 'run'+str(run)+'/structure_database/xyz_files/'+str(p_hash)+'.xyz'
                p_file_name = tar.extractfile(p_file_name)
                p_file = p_file_name.read().decode('utf-8')
                p_lines = p_file.split('\n')
                for smiles, lines, idx in zip([r_canonical, p_canonical], [r_lines, p_lines], [n_reactant, n_product]):
                    energy=None
                    for line in lines:
                        if 'SCF done' in line:
                            energy = line.split()[2]
                            #print(energy)
                    if energy:
                        COMPLEX_ENERGIES.loc[(smiles, idx), 'energy'] = energy
    os.chdir('../')


if __name__ == "__main__":
    SMILES_DF = pd.read_csv(sys.argv[1], index_col=0)
    MULT_INDEX = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['smiles', 'index'])
    COMPLEX_ENERGIES = pd.DataFrame(index=MULT_INDEX)
    for i in SMILES_DF.index:
        print(i)
        get_complex_energies(str(i))
    print(COMPLEX_ENERGIES)
    COMPLEX_ENERGIES.to_csv('complex_energies.csv')
