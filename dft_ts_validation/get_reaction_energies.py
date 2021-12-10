#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python

import sys
import os
import subprocess

import numpy as np
import pandas as pd

from run_dft_ts import get_smiles


def optimize_structures(path):
    pwd = os.getcwd()
    os.chdir(path)

    opt_files = [f for f in os.listdir(os.curdir) if f.endswith('opt.xyz')]
    print("# of opt file:", len(opt_files))
    #if opt_files:
    #    sys.exit("optimization allready done")

    print("doing optimizations")
    xyz_files = [f for f in os.listdir(os.curdir) if f.endswith('forward.xyz') or f.endswith('reverse.xyz')]
    for xyz_file in xyz_files:
        print(xyz_file)
        com_file = write_opt_com_file(xyz_file, 2, 4)
        output = run_cmd("/groups/kemi/mharris/bin/submit_g16 {0}".format(com_file))
        #_ = calc_gaussian(write_opt_com_file, xyz_file, 2, 4)
    os.chdir(pwd)


def get_dft_path(reactant, r_idx, letter, index):
    path_dft = reactant+'_'+r_idx+'_'+letter+'_dft'
    path = os.path.join(reactant, path_dft)
    path = os.path.join(path, index)
    path = os.path.join(path, 'ts_test_dft')
    path = os.path.join(path, 'dft_interpolation')

    print(path)
    return path


def extract_optimized_structure(out_file, n_atoms, atom_labels):
    """
    After waiting for the constrained optimization to finish, the
    resulting structure from the constrained optimization is
    extracted and saved as .xyz file ready for TS optimization.
    """
    optimized_xyz_file = out_file[:-4]+".xyz"
    with open(out_file, 'r') as ofile:
        line = ofile.readline()
        while line:
            if 'Standard orientation' in line or 'Input orientation' in line:
                coordinates = np.zeros((n_atoms, 3))
                for i in range(5):
                    line = ofile.readline()
                for i in range(n_atoms):
                    coordinates[i, :] = np.array(line.split()[-3:])
                    line = ofile.readline()
            line = ofile.readline()
    with open(optimized_xyz_file, 'w') as _file:
        _file.write(str(n_atoms)+'\n\n')
        for i in range(n_atoms):
            _file.write(atom_labels[i])
            for j in range(3):
                _file.write(' '+"{:.5f}".format(coordinates[i, j]))
            _file.write('\n')


    return optimized_xyz_file


def atom_information(xyz_file):
    """
    extract information about system: number of atoms and atom numbers
    """
    atom_numbers = []
    with open(xyz_file, 'r') as _file:
        line = _file.readline()
        n_atoms = int(line.split()[0])
        _file.readline()
        for i in range(n_atoms):
            line = _file.readline().split()
            atom_number = line[0]
            atom_numbers.append(atom_number)

    return n_atoms, atom_numbers


def extract_energies(out_file):
    E = np.nan
    E_0 = np.nan
    with open(out_file, 'r') as _file:
        line = _file.readline()
        while line:
            if 'SCF Done' in line:
                E = line.split()[4]
            if 'zero-point' in line:
                E_0 = line.split()[6]
            line = _file.readline()

    return E, E_0


def get_energies(path):
    pwd = os.getcwd()
    os.chdir(path)
    ts_file = [f for f in os.listdir(os.curdir) if f.endswith('ts.out')][0]
    try:
        n_atoms, atom_labels = atom_information(ts_file[:-4]+'.xyz')
    except FileNotFoundError:
        print('TS optimization crashed')
        os.chdir(pwd)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    ts_e, ts_e_0 = extract_energies(ts_file)
    rev_file = [f for f in os.listdir(os.curdir) if f.endswith('reverse_opt2_freq.out')][0]
    for_file = [f for f in os.listdir(os.curdir) if f.endswith('forward_opt2_freq.out')][0]
    rev_e, rev_e_0 = extract_energies(rev_file)
    for_e, for_e_0 = extract_energies(for_file)

    rev_xyz = extract_optimized_structure(rev_file, n_atoms, atom_labels)
    for_xyz = extract_optimized_structure(for_file, n_atoms, atom_labels)

    rev_smiles, _, _ = get_smiles(rev_xyz, 0)
    for_smiles, _, _ = get_smiles(for_xyz, 0)

    os.chdir(pwd)

    return ts_e, ts_e_0, rev_e, rev_e_0, for_e, for_e_0, rev_smiles, for_smiles


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)
    #reactant = sys.argv[2]
    #r_idx = sys.argv[3]
    #letter = sys.argv[4]

    for reactant, r_idx, letter, index in zip(df.reactant, df.r_idx, df.letter,
                                              df.index):
        print(index)
        path = get_dft_path(str(reactant), str(r_idx), str(letter), str(index))
        ts_e, ts_e_0, rev_e, rev_e_0, for_e, for_e_0, rev_smiles, for_smiles = get_energies(path)
        df.loc[index, 'ts_e'] = ts_e
        df.loc[index, 'ts_e_0'] = ts_e_0
        df.loc[index, 'rev_e'] = rev_e
        df.loc[index, 'rev_e_0'] = rev_e_0
        df.loc[index, 'for_e'] = for_e
        df.loc[index, 'for_e_0'] = for_e_0
        df.loc[index, 'rev_smiles'] = rev_smiles
        df.loc[index, 'for_smiles'] = for_smiles
        print(ts_e, ts_e_0, rev_e, rev_e_0, for_e, for_e_0, rev_smiles, for_smiles)
    print(df)
    df.to_csv(sys.argv[1][:-4]+'_upd.csv')
