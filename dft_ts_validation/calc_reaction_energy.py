#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python

import sys
import os
import subprocess
print(subprocess.__file__)
import pandas as pd
import numpy as np

def run_cmd(cmd):
    """
    run command line
    """
    cmd = cmd.split()
    print(cmd)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    return output.decode('utf-8')


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


def write_opt_com_file(xyz_file, cpus, mem):
    """ prepares com file for gaussian """

    com_file = xyz_file[:-4]+'_freq.com'
    with open(com_file, 'w') as _file:
        _file.write('%nprocshared='+str(cpus)+'\n')
        _file.write('%mem='+str(mem)+'GB'+'\n')
        _file.write('#freq UwB97XD/Def2TZVP \n\n')
        _file.write('something title\n\n')
        _file.write('0 1\n')
        with open(xyz_file, 'r') as file_in:
            lines = file_in.readlines()[2:]
            _file.writelines(lines)
        _file.write('\n')

    return com_file


def optimize_structures(path):
    pwd = os.getcwd()
    os.chdir(path)

    out_files = [f for f in os.listdir(os.curdir) if f.endswith('opt2.out')]

    if not out_files:
        os.chdir(pwd)
        return

    n_atoms, atom_numbers = atom_information(out_files[0][:-9]+'.xyz')
    print("doing optimizations")
    #xyz_files = [f for f in os.listdir(os.curdir) if f.endswith('forward.xyz') or f.endswith('reverse.xyz')]
    for out_file in out_files:
        xyz_file = extract_optimized_structure(out_file, n_atoms, atom_numbers)
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


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)

    for reactant, r_idx, letter, index in zip(df.reactant, df.r_idx, df.letter,
                                              df.index):
        path = get_dft_path(str(reactant), str(r_idx), str(letter), str(index))
        optimize_structures(path)
