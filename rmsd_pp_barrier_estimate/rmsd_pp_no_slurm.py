#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python

import os
import sys
import subprocess
import shutil
import textwrap
import itertools


import numpy as np
import pandas as pd

import xyz2mol_local

from math import isnan
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdmolops
from rdkit.Chem import rdchem
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Geometry import Point3D


def run_cmd(cmd):
    """
    Run command line
    """
    cmd = cmd.split()
    #print(cmd)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    return output.decode('utf-8')


def reorder_atoms_to_map(mol):

    """
    Reorders the atoms in a mol objective to match that of the mapping
    """

    atom_map_order = np.zeros(mol.GetNumAtoms()).astype(int)
    for atom in mol.GetAtoms():
        map_number = atom.GetAtomMapNum()-1
        atom_map_order[map_number] = atom.GetIdx()
    mol = Chem.RenumberAtoms(mol, atom_map_order.tolist())
    return mol


def bonds_getting_formed_or_broken(rsmi, psmi, n_atoms):
    """
    Based on the reaction and product structure, the bonds that are
    fomed/broken are singled out for contraintment
    the difference in the afjacency matric tells whether bond has been formed
    (+1) or bond is broken (-1)
    """

    bond_pairs_changed = []
    rmol = Chem.MolFromSmiles(rsmi, sanitize=False)
    pmol = Chem.MolFromSmiles(psmi, sanitize=False)
    rmol = reorder_atoms_to_map(rmol)
    pmol = reorder_atoms_to_map(pmol)
    Chem.SanitizeMol(rmol)
    Chem.SanitizeMol(pmol)

    p_ac = Chem.rdmolops.GetAdjacencyMatrix(pmol)
    r_ac = Chem.rdmolops.GetAdjacencyMatrix(rmol)

    difference_mat = p_ac - r_ac
    for combination in itertools.combinations(range(n_atoms), 2):
        combination = list(combination)
        bond_change = difference_mat[combination[0], combination[1]]
        if bond_change != 0:
            bond_pairs_changed.append(combination)

    return bond_pairs_changed


def check_activity_of_bonds(xyz_file, bond_pairs):
    """
    This function checks whether the bonds being formed or broken fulfills the
    "activity" criteria that 1.2 =< r_ij/(r_cov,i+r_cov,j) =< 1.7
    where r_ij is the bond distance of bond pair i,j in the structure in
    xyz_file. r_cov,i is the covalent distance of atom i. This criteria is
    taken from atom_mapper paper.
    """
    active_bonds = []
    atom_numbers_list, coordinates_list, _ = get_coordinates([xyz_file])
    ptable = Chem.GetPeriodicTable()
    atom_numbers, coordinates = atom_numbers_list[0], coordinates_list[0]

    for bond_pair in bond_pairs:
        atom_i = atom_numbers[bond_pair[0]]
        atom_j = atom_numbers[bond_pair[1]]
        #print(atom_i, atom_j)
        r_distance =  \
        np.linalg.norm(coordinates[bond_pair[0], :]-coordinates[bond_pair[1], :])
        r_cov_i = ptable.GetRcovalent(atom_i)
        r_cov_j = ptable.GetRcovalent(atom_j)
        bond_activity = r_distance/(r_cov_i+r_cov_j)
        #print('Bond activity for ', bond_pair, ' = '+str(bond_activity))
        if 1.2 <= bond_activity <= 1.7:
            print("bond active")
            active_bonds.append(True)
        else:
            print("bond not active")
            active_bonds.append(False)
    return active_bonds


def write_path_input(product_xyz, k_push, k_pull, alp, temp):
    """
    Write the input file for a path xtb calculation based on the input
    variables
    """
    path_file = """\
    $path
       nopt=100
       anopt=3
       kpush={0}
       kpull={1}
       alp={2}
       product={3}
    $end
    $scc
       temp={4}
    $end
    $opt
       optlevel=2
    $end
    """.format(k_push, k_pull, alp, product_xyz, temp)

    with open('path.inp', 'w') as ofile:
        ofile.write(textwrap.dedent(path_file))


def do_xtb_path_calculation(reactant_xyz, product_xyz, k_push, k_pull, alp,
                            temp, charge, cpus=1, mem=2):
    """
    Change the path.inp file according to parameters, and run the path
    calculation
    """
    job_name = reactant_xyz[:-4]+'_'+product_xyz[:-4]
    os.mkdir(job_name)
    shutil.copy(reactant_xyz, job_name)
    shutil.copy(product_xyz, job_name)
    os.chdir(job_name)

    write_path_input(product_xyz, k_push, k_pull, alp, temp)

    os.environ["XTBHOME"] = "/groups/kemi/koerstz/opt/xtb/6.1/bin"

    os.environ["OMP_STACKSIZE"] = str(mem)+'G'
    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)

    # output = run_cmd('ulimit -s unlimited')
    os.system('ulimit -s unlimited')
    output = run_cmd("/groups/kemi/koerstz/opt/xtb/6.1/bin/xtb {0} --path --input path.inp --gfn2 --chrg {1}".format(reactant_xyz, charge))
    with open(job_name+'.out', 'w') as _file:
        _file.write(output)
    os.chdir('../')

    outfile = job_name+'/'+job_name+'.out'
    return outfile


def check_if_reaction_complete(out_file):
    """
    From xtb path calculation where three runs have been done. Based on the
    RMSE between end structure and target structure it is determined whether
    the reaction has been completed.
    Returns list of "True" and "False" for the three rections depending on
    whether the reacion was finished or not, respectively.
    TODO: need no include strategy when path calculation does not converge (see
    e.g. reaction 35)
    """
    barriers = []
    rmse_prod_to_endpath = []
    reactions_completed = []
    if os.path.isfile(out_file.split('/')[0]+'/'+"NOT_CONVERGED"):
        print("path job crashed")
        return [np.nan, np.nan, np.nan]
    with open(out_file, 'r') as _file:
        line = _file.readline()
        while line:
            if "energies in kcal/mol, RMSD in Bohr" in line:
                for _ in range(3):
                    line = _file.readline()
                    data = line.split()
                    if data[0] == "WARNING:":
                        line = _file.readline()
                        data = line.split()
                    try:
                        barriers.append(np.float(data[3]))
                        rmsd = np.float(data[-1])
                        rmse_prod_to_endpath.append(rmsd)
                        if rmsd < 0.1:  # lowered the rmsd demand a bit
                            reactions_completed.append(True)
                        else:
                            reactions_completed.append(False)
                    except:
                        reactions_completed.append(np.nan)
            if "GEOMETRY OPTIMIZATION FAILED!" in line:
                print("path job crashed (file)")
                return [np.nan, np.nan, np.nan]
            line = _file.readline()
    print(reactions_completed)
    return reactions_completed


def get_relaxed_xtb_structure(xyz_path, new_file_name):
    """
    choose the relaxed structure from the last point on the xtb path
    """
    with open(xyz_path, 'r') as _file:
        line = _file.readline()
        n_lines = int(line.split()[0])+2
    count = 0
    input_file = open(xyz_path, 'r')
    dest = None
    for line in input_file:
        if count % n_lines == 0:
            if dest:
                dest.close()
            dest = open(new_file_name, "w")
        count += 1
        dest.write(line)


def chiral_tags(mol):
    """
    Tag methylene and methyl groups with a chiral tag priority defined
    from the atom index of the hydrogens
    """
    li_list = []
    smarts_ch2 = '[!#1][*]([#1])([#1])([!#1])'
    atom_sets = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_ch2))
    for atoms in atom_sets:
        atoms = sorted(atoms[2:4])
        prioritized_H = atoms[-1]
        li_list.append(prioritized_H)
        mol.GetAtoms()[prioritized_H].SetAtomicNum(9)
    smarts_ch3 = '[!#1][*]([#1])([#1])([#1])'
    atom_sets = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_ch3))
    for atoms in atom_sets:
        atoms = sorted(atoms[2:])
        H1 = atoms[-1]
        H2 = atoms[-2]
        li_list.append(H1)
        li_list.append(H2)
        mol.GetAtoms()[H1].SetAtomicNum(9)
        mol.GetAtoms()[H2].SetAtomicNum(9)

    Chem.AssignAtomChiralTagsFromStructure(mol, -1)
    rdmolops.AssignStereochemistry(mol)
    for atom_idx in li_list:
        mol.GetAtoms()[atom_idx].SetAtomicNum(1)

    return mol


def choose_resonance_structure(mol):
    """
    This function creates all resonance structures of the mol object, counts
    the number of rotatable bonds for each structure and chooses the one with
    fewest rotatable bonds (most 'locked' structure)
    """
    resonance_mols = rdchem.ResonanceMolSupplier(mol,
                                                 rdchem.ResonanceFlags.ALLOW_CHARGE_SEPARATION)
    res_status = True
    new_mol = None
    if not resonance_mols:
        print("using input mol")
        new_mol = mol
        res_status = False
    for res_mol in resonance_mols:
        Chem.SanitizeMol(res_mol)
        n_rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(res_mol)
        if new_mol is None:
            smallest_rot_bonds = n_rot_bonds
            new_mol = res_mol
        if n_rot_bonds < smallest_rot_bonds:
            smallest_rot_bonds = n_rot_bonds
            new_mol = res_mol

    Chem.DetectBondStereochemistry(new_mol, -1)
    rdmolops.AssignStereochemistry(new_mol, flagPossibleStereoCenters=True,
                                   force=True)
    Chem.AssignAtomChiralTagsFromStructure(new_mol, -1)
    return new_mol, res_status


def extract_smiles(xyz_file, charge, allow_charge=True):
    """
    uses xyz2mol to extract smiles with as much 3d structural information as
    possible
    """
    atoms, _, xyz_coordinates = xyz2mol_local.read_xyz_file(xyz_file)
    
    try:
        input_mol = xyz2mol_local.xyz2mol(atoms, xyz_coordinates, charge=charge,
                                          use_graph=True,
                                          allow_charged_fragments=allow_charge,
                                          use_huckel=True, use_atom_maps=True,
                                          embed_chiral=True)
    except:
        input_mol = xyz2mol_local.xyz2mol(atoms, xyz_coordinates, charge=charge,
                                          use_graph=True,
                                          allow_charged_fragments=allow_charge,
                                          use_huckel=False, use_atom_maps=True,
                                          embed_chiral=True)

    input_mol = reorder_atoms_to_map(input_mol)
    structure_mol, res_status = choose_resonance_structure(input_mol)
    structure_mol = chiral_tags(structure_mol)
    rdmolops.AssignStereochemistry(structure_mol)
    structure_smiles = Chem.MolToSmiles(structure_mol)

    return structure_smiles, GetFormalCharge(structure_mol), res_status


def get_smiles(xyz_file, charge):
    """
    Try different things to extract sensible smiles using xyz2mol
    """
    
    
    try:
        smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                           allow_charge=True)
        #print(smiles)
        if formal_charge != charge:
            smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                               allow_charge=False)
    except:
        try:
            smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                               allow_charge=False)
        except:
            return None, None, None

    return smiles, formal_charge, res_status


def get_reactant_product_smiles(reactant_file, product_file, charge):
    """
    Extract atom mapped smiles of the intended reactant and product
    """
    # extract reactant smiles
    reactant_smiles, _, _ = get_smiles(reactant_file, charge)
    #print(reactant_smiles)

    # extract product smiles
    product_smiles, _, _ = get_smiles(product_file, charge)
    #print(product_smiles)

    return [reactant_smiles, product_smiles]


def check_smiles_of_endpoints(input_xyz, output_xyz, smiles_list, charge,
                              forward):
    """
    check with xyz2mol if reaction to intermediate occured
    """
    # extract in and ouput smiles
    reaction_found = False
    smiles = get_reactant_product_smiles(input_xyz, output_xyz, charge)
    input_smiles, output_smiles = smiles[0], smiles[1]
    new_smiles_list = smiles_list.copy()

    print("start smiles:", input_smiles)
    print("end smiles:", output_smiles)

    input_idx = smiles_list.index(input_smiles)
    if forward:
        intermediate_idx = input_idx+1
    else:
        intermediate_idx = input_idx
    print(input_idx, forward)
    if output_smiles not in smiles_list:
        print("new structure encountered")
        new_smiles_list.insert(intermediate_idx, output_smiles)
        print(new_smiles_list, smiles_list)

    if (forward and output_smiles == smiles_list[-1]) or (not forward and output_smiles == smiles_list[0]):
        reaction_found = True
        print("Warning: reaction found (smiles match) but RMSD > 0.5")

    return new_smiles_list, reaction_found


def extract_xtb_structures(path_file, directory, n_atoms):
    """
    Based on the structure path predicted by xtb: do xtb single point energies
    for each structure
    """

    os.mkdir(directory)
    n_lines = n_atoms+2

    count = 0
    indx = 0
    dest = None
    with open(path_file, 'r') as _file:
        for line in _file:
            if count % n_lines == 0:
                if dest:
                    dest.close()
                dest = open(directory+'/'+str(indx)+'.xyz', 'w')
                indx += 1
            dest.write(line)
            count += 1
        dest.close()


def get_coordinates(structure_files_list):
    """
    Extrapolate around maximum structure on the xtb surface to make DFT single
    point calculations in order to choose the best starting point for TS
    optimization. Should return this starting point structure
    """
    n_structures = len(structure_files_list)
    atom_numbers_list = []
    coordinates_list = []
    for i in range(n_structures):
        atom_numbers = []
        with open(structure_files_list[i], 'r') as struc_file:
            line = struc_file.readline()
            n_atoms = int(line.split()[0])
            struc_file.readline()
            coordinates = np.zeros((n_atoms, 3))
            for j in range(n_atoms):
                line = struc_file.readline().split()
                atom_number = line[0]
                atom_numbers.append(atom_number)
                coordinates[j, :] = np.array([np.float(num) for num in
                                              line[1:]])
        atom_numbers_list.append(atom_numbers)
        coordinates_list.append(coordinates)
    return atom_numbers_list, coordinates_list, n_atoms


def make_sp_extrapolation(atom_numbers_list, coordinates_list, n_atoms,
                          n_points, directory):
    """
    From the given structures in coordinates_list xyz files are created by
    extrapolating between those structures with n_points between each structure
    creates a directory "path" with those .xyz files
    """
    n_structures = len(coordinates_list)
    os.mkdir(directory)
    with open(directory+'/path_file.txt', 'w') as path_file:
        for i in range(n_structures-1):
            difference_mat = coordinates_list[i+1]-coordinates_list[i]
            for j in range(n_points+1):
                path_xyz = coordinates_list[i]+j/n_points*difference_mat
                path_xyz = np.matrix(path_xyz)
                file_path = directory+'/path_point_' + str(i*n_points+j)+'.xyz'
                with open(file_path, 'w+') as _file:
                    _file.write(str(n_atoms)+'\n\n')
                    path_file.write(str(n_atoms)+'\n\n')
                    for atom_number, line in zip(atom_numbers_list[i], path_xyz):
                        _file.write(atom_number+' ')
                        path_file.write(atom_number+' ')
                        np.savetxt(_file, line, fmt='%.6f')
                        np.savetxt(path_file, line, fmt='%.6f')


def find_xtb_max_from_sp_extrapolation(directory, extract_max_structures):
    """
    when sp calculations ar efinished: find the structure with maximum xtb
    energy
    """
    energies = []
    path_points = []

    os.chdir(directory)

    files = [f for f in os.listdir(os.curdir) if
             f.endswith("xtbout")]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for file_name in files[1:-1]:
        path_point = int(''.join(filter(str.isdigit, file_name)))
        path_points.append(path_point)
        with open(file_name, 'r') as _file:
            line = _file.readline()
            while line:
                if 'TOTAL ENERGY' in line:
                    energy_au = np.float(line.split()[3])
                line = _file.readline()
        energies.append(energy_au)
        # os.remove(file_name)
    energies_kcal = np.array(energies)*627.509
    energies_kcal = energies_kcal-energies_kcal[0]
    max_index = energies.index(max(energies))
    if extract_max_structures:
        max_point = path_points[max_index]
        shutil.copy(str(max_point-1)+'.xyz', '../maxE-1.xyz')
        shutil.copy(str(max_point)+'.xyz', '../maxE.xyz')
        shutil.copy(str(max_point+1)+'.xyz', '../maxE+1.xyz')
    os.chdir("../")
    files = ["maxE-1.xyz", "maxE.xyz", "maxE+1.xyz"]
    return max(energies), energies_kcal, max_index, files


def find_max_AB_structure(directory, bond_pairs, energies):
    """
    goes through all structures in the path and for each structure determines
    how many of the bonds that break/form during the reaction are active.
    returns structure+neighbour structures with  maximum active bonds.
    If multiple structures with same # of active bonds: return maximum energy
    structure.
    """
    path_points = []
    bonds_active = []
    os.chdir(directory)

    files = [f for f in os.listdir(os.curdir) if f.endswith('.xyz')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for xyz_file in files[1:-1]:
        path_point = int(''.join(filter(str.isdigit, xyz_file)))
        path_points.append(path_point)
        bond_activity = check_activity_of_bonds(xyz_file, bond_pairs)
        n_bonds_active = np.count_nonzero(bond_activity)
        bonds_active.append(n_bonds_active)

    max_AB = max(bonds_active)
    max_AB_idx = [i for i, j in enumerate(bonds_active) if j == max_AB]

    e_subset = [energies[i] for i in max_AB_idx] 
    max_idx = e_subset.index(max(e_subset))
    path_idx = path_points[max_AB_idx[max_idx]]

    shutil.copy(str(path_idx-1)+'.xyz', '../maxAB-1.xyz')
    shutil.copy(str(path_idx)+'.xyz', '../maxAB.xyz')
    shutil.copy(str(path_idx+1)+'.xyz', '../maxAB+1.xyz')

    os.chdir('../')

    files = ["maxAB-1.xyz", "maxAB.xyz", "maxAB+1.xyz"]

    return max_AB, path_idx, files


def get_bond_activity(xyz_file, bond_pairs):
    """
    for each of the bonds forming/breakingduring the reaction, label the bond
    with
    0 if the bond is broken
    1 if the bond is active
    2 if the bond is formed
    """
    active_bonds = np.zeros(len(bond_pairs))
    atom_numbers_list, coordinates_list, _ = get_coordinates([xyz_file])
    ptable = Chem.GetPeriodicTable()
    atom_numbers, coordinates = atom_numbers_list[0], coordinates_list[0]

    for i, bond_pair in enumerate(bond_pairs):
        atom_i = atom_numbers[bond_pair[0]]
        atom_j = atom_numbers[bond_pair[1]]
    #    print(atom_i, atom_j)
        r_distance =  \
            np.linalg.norm(coordinates[bond_pair[0], :]-coordinates[bond_pair[1], :])
        r_cov_i = ptable.GetRcovalent(atom_i)
        r_cov_j = ptable.GetRcovalent(atom_j)
        bond_activity = r_distance/(r_cov_i+r_cov_j)
        if bond_activity < 1.2:
            # unbound
            active_bonds[i] = 0
        elif 1.2 <= bond_activity <= 1.7:
            # active
            active_bonds[i] = 1
        elif 1.7 <= bond_activity:
            # bound
            active_bonds[i] = 2
        else:
            print("something's wrong....")
            active_bonds.append(np.nan)

    return active_bonds


def find_max_BC_structure(directory, bond_pairs, energies):
    """
    goes through all structures in the path and for each pair of structures
    connected by two steps calculates the number of changes in the adjacency
    matrices. The pair that corresponds to maximum changes is chosen. If
    multiple pairs are equal, the pair with maximum energy for the middle
    structure is chosen. Then returns the middle path index number and extracts
    the three structures ready for interpolation.
    """
    path_points = []
    bond_changes = []
    os.chdir(directory)
    files = [f for f in os.listdir(os.curdir) if f.endswith('.xyz')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for i, xyz_file in zip(np.arange(len(files)-2)+1, files[1:-1]):
        path_point = int(''.join(filter(str.isdigit, xyz_file)))
        print(path_point)
        path_points.append(path_point)
        bond_activity1 = get_bond_activity(files[i-1], bond_pairs)
        print(bond_activity1)
        bond_activity2 = get_bond_activity(files[i+1], bond_pairs)
        print(bond_activity2)
        bond_change = np.sum(np.abs(bond_activity2-bond_activity1))
        print(bond_change)
        bond_changes.append(bond_change)

    max_BC = max(bond_changes)
    max_BC_idx = [i for i, j in enumerate(bond_changes) if j == max_BC]

    e_subset = [energies[i] for i in max_BC_idx]
    max_idx = e_subset.index(max(e_subset))
    path_idx = path_points[max_BC_idx[max_idx]]
    if path_idx == 0:
        path_idx += 1
    shutil.copy(str(path_idx-1)+'.xyz', '../maxBC-1.xyz')
    shutil.copy(str(path_idx)+'.xyz', '../maxBC.xyz')
    shutil.copy(str(path_idx+1)+'.xyz', '../maxBC+1.xyz')

    os.chdir('../')

    files = ["maxBC-1.xyz", "maxBC.xyz", "maxBC+1.xyz"]

    return max_BC, path_idx, files


def try_xtb_path(reactant_file, product_file, kpull, kpush, alpha, temp, charge):
    """
    Trying an xtb path and checks if reaction completed
    """
    outfile = do_xtb_path_calculation(reactant_file, product_file, kpull,
                                      kpush, alpha, temp, charge)
    reactions_completed_boolean = check_if_reaction_complete(outfile)
    if True in reactions_completed_boolean:
        path_file = reactant_file[:-4] + '_' + product_file[:-4]+'/xtbpath_' \
                    + str(reactions_completed_boolean.index(True)+1) + '.xyz'
        return [path_file, outfile, reactions_completed_boolean.index(True)]

    try:
        not_nan = next(x for x in reactions_completed_boolean if not isnan(x))
        not_nan_idx = reactions_completed_boolean.index(not_nan)
        path_file = reactant_file[:-4]+'_'+product_file[:-4]+'/xtbpath_'+str(not_nan_idx+1)+'.xyz'
        print(path_file)
        reactant_file = reactant_file[:-4]+'_rel.xyz'
        get_relaxed_xtb_structure(path_file, reactant_file)

    except StopIteration:
        # if all path searches fails use input reactant structure
        shutil.copy(reactant_file, reactant_file[:-4]+'_rel.xyz')
        reactant_file = reactant_file[:-4]+'_rel.xyz'
        path_file = None
        not_nan_idx = np.nan


    return [reactant_file, product_file, path_file, outfile, not_nan_idx]


def find_xtb_path(reactant_file, product_file, temp, charge=0):
    """
    Find an xtb path combining the reactant and product file
    """
    smiles_list = get_reactant_product_smiles(reactant_file, product_file,
                                              charge)
    print(smiles_list)
    forward = True
    path_files_list = []
    out_files_list = []
    path_index_list = []
    # n_structures = len(smiles_list)
    kpull_list = [-0.02, -0.02, -0.02, -0.03, -0.03, -0.04, -0.04]
    alp_list = [0.6, 0.3, 0.3, 0.6, 0.6, 0.6, 0.4]
    #output_list = try_xtb_path(reactant_file, product_file, 0.008,  #lowered initial push
    #                           kpull_list[0], alp_list[0], temp)
    #i = 0
    #while len(output_list) != 3:
    #    i += 1
    #    if i == 7:
    #        print("xtb path not found")
    #        path_files_list.append(output_list[0])
    #        out_files_list.append(output_list[1])
    #        path_index_list.append(None)
    #        return tuple([path_files_list, out_files_list, path_index_list])

    #    old_length_smiles_list = len(smiles_list)
    #    print("old:", len(smiles_list))
    #    new_smiles_list, reaction_found = check_smiles_of_endpoints(reactant_file, output_list[0],
    #                                                                smiles_list, charge, forward)
    #    new_length_smiles_list = len(new_smiles_list)
    #    print("new:", len(new_smiles_list))
    #    print(len(new_smiles_list), len(smiles_list))
    #    if old_length_smiles_list != new_length_smiles_list:
    #        print("found a possible intermediate structure")
    #        path_files_list.append(output_list[2])
    #        out_files_list.append(output_list[3])
    #        print(out_files_list)
    #        print("idx:", output_list[4])
    #        path_index_list.append(output_list[4])
    #        print(path_index_list)

    #    print(new_smiles_list)
    #    reactant_file = output_list[1]
    #    product_file = output_list[0]
    #    output_list = try_xtb_path(reactant_file, product_file, 0.01,
    #                               kpull_list[i], alp_list[i], temp)
    #    smiles_list = new_smiles_list.copy()

    #    #change direction
    #    forward = not forward

    kpush = 0.008  # initial kpush
    for kpull, alp in zip(kpull_list, alp_list):
        output_list = try_xtb_path(reactant_file, product_file, kpush,
                                   kpull, alp, temp, charge)
        if len(output_list) == 3:
            path_files_list.append(output_list[0])
            print(path_files_list)
            out_files_list.append(output_list[1])
            path_index_list.append(output_list[2])
            return tuple([path_files_list, out_files_list, path_index_list])

        new_smiles_list, reaction_found = check_smiles_of_endpoints(reactant_file, output_list[0],
                                                                    smiles_list, charge, forward)
        print(reaction_found, len(new_smiles_list), len(smiles_list))
        if reaction_found or len(new_smiles_list) != len(smiles_list):
            path_files_list.append(output_list[2])
            out_files_list.append(output_list[3])
            path_index_list.append(output_list[4])
            print(path_files_list)

        if reaction_found:
            return tuple([path_files_list, out_files_list, path_index_list])

        smiles_list = new_smiles_list.copy()
        kpush = 0.01

        reactant_file = output_list[1]
        product_file = output_list[0]

        # change direction
        forward = not forward

    #if len(output_list) == 3:
    #    path_files_list.append(output_list[0])
    #    out_files_list.append(output_list[1])
    #    path_index_list.append(output_list[2])
    print("xtb path not found")

    return tuple([path_files_list, out_files_list, None])


def write_xyz_file(mol, file_name):
    """
    write xyz file with cooridnates optimized w ff based on input smiles
    """
    n_atoms = mol.GetNumAtoms()
    charge = Chem.GetFormalCharge(mol)

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    with open(file_name, 'w') as _file:
        _file.write(str(n_atoms)+'\n\n')
        for atom, symbol in enumerate(symbols):
            coord = mol.GetConformers()[0].GetAtomPosition(atom)
            line = " ".join((symbol, str(coord.x), str(coord.y), str(coord.z),
                             "\n"))
            _file.write(line)
        if charge != 0:
            _file.write("$set\n")
            _file.write("chrg "+str(charge)+"\n")
            _file.write("$end")

    return mol


def xtb_path_parameter(n_path, out_file, _dict):
    """
    This function extracts the kpull, kpush and alp values used in the
    succesful xtb path
    """
    push_pull_list = []
    with open(out_file, 'r') as _file:
        line = _file.readline()
        while line:
            if 'Gaussian width (1/Bohr)' in line:
                alp = line.split()[4]
                _dict.update({"ALP [1/a0]": alp})
            if 'actual k push/pull:' in line:
                push_pull_list.append(line)
            line = _file.readline()
    k_push_pull = push_pull_list[n_path].split()
    _dict.update({"k_push": np.float(k_push_pull[4]), "k_pull":
                  np.float(k_push_pull[5])})
    return _dict


def do_rmsd_pp_calc(reactant_file, product_file, n_atoms, reaction_index,
                    bond_pairs, charge=0):
    """
    submits and waits for rmsd_pp path calculation to finish
    """
    high_temperature = False

    #write to .xyz files
    #product_file = 'reactant.xyz'
    #reactant_file = 'product.xyz'

    #n_atoms = r_mol.GetNumAtoms()

    #r_mol = write_xyz_file(r_mol, reactant_file)
    #p_mol = write_xyz_file(p_mol, product_file)

    path_files, outfiles, n_paths = find_xtb_path(reactant_file, product_file,
                                                  temp=300, charge=charge)
    print(path_files, outfiles, n_paths)
    if not n_paths or np.isnan(n_paths).any():
        os.mkdir("ht")
        shutil.copy(reactant_file, "ht")
        shutil.copy(product_file, "ht")
        os.chdir("ht")
        path_files, outfiles, n_paths = find_xtb_path(reactant_file,
                                                      product_file, temp=6000,
                                                      charge=charge)
        high_temperature = True
        if not n_paths or np.isnan(n_paths).any():
            print("no path could be found")
            os.chdir('../../')
            return [[np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan]]
    max_xtb_energies = []
    interpolation_points = []
    i = 0
    for path_file, outfile, n_path in zip(path_files, outfiles, n_paths):
        #_dict = xtb_path_parameter(n_path, outfile, _dict)
        extract_xtb_structures(path_file, "xtb_sp"+str(i), n_atoms)
        os.chdir("xtb_sp"+str(i))
        xyz_files = [f for f in os.listdir() if f.endswith('.xyz')]
        for xyz_file in xyz_files:
            output = run_cmd("/groups/kemi/koerstz/opt/xtb/6.1/bin/xtb {0} --gfn2 --acc \
            0.2 --chrg {1}".format(xyz_file, charge))
            with open(xyz_file[:-4]+'.xtbout', 'w') as _file:
                _file.write(output)
        os.chdir('../')
        # interpolation structures based on max energy
        max_energy_sp, xtb_sp_energies, E_idx, maxE_files = \
            find_xtb_max_from_sp_extrapolation("xtb_sp"+str(i), True)

        # interpolation structures based on max number of active bonds
        maxAB, AB_idx, maxAB_files = find_max_AB_structure("xtb_sp"+str(i),
                                                           bond_pairs,
                                                           xtb_sp_energies)
        # interpolation structures based on max change in adjacency matrix
        maxCB, BC_idx, maxBC_files = find_max_BC_structure("xtb_sp"+str(i),
                                                           bond_pairs,
                                                           xtb_sp_energies)
        int_points = [E_idx, AB_idx, BC_idx]
        interpolation_points.append(int_points)
        #files_list = ['max_structure-1.xyz', 'max_structure.xyz',
        #              'max_structure+1.xyz']
        max_int_energies = []
        for files_list, int_dir in zip([maxE_files, maxAB_files, maxBC_files],
                                       ['pathE', 'pathAB', 'pathBC']):
            atom_numbers, coordinates_list, n_atoms = get_coordinates(files_list)
            n_points = 10
            make_sp_extrapolation(atom_numbers, coordinates_list, n_atoms,
                                  n_points, int_dir+str(i))
            os.chdir(int_dir+str(i))
            xyz_files = [f for f in os.listdir() if f.endswith('.xyz')]
            for xyz_file in xyz_files:
                output = run_cmd("/groups/kemi/koerstz/opt/xtb/6.1/bin/xtb {0} --gfn2 --acc \
                0.2 --chrg {1}".format(xyz_file, charge))
                with open(xyz_file[:-4]+'.xtbout', 'w') as _file:
                    _file.write(output)
            os.chdir('../')

            max_xtb_energy, xtb_sp_energies_kcal, max_energy, _ =\
                find_xtb_max_from_sp_extrapolation(int_dir+str(i), False)
            max_int_energies.append(max_xtb_energy)
        max_xtb_energies.append(max_int_energies)
        i += 1

    os.chdir('../')
    if high_temperature:
        os.chdir('../')

    return max_xtb_energies, interpolation_points


def embed_smiles_far(smiles):
    """
    create 3D conformer with atom order matching atom mapping. If more than one
    fragment present, they are moved 0.5*n_atoms from each other
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = reorder_atoms_to_map(mol)

    fragments = Chem.GetMolFrags(mol, asMols=True)
    n_atoms = mol.GetNumAtoms()
    coordinates = np.zeros((n_atoms, 3))

    for n_frag, fragment in enumerate(fragments):
        Chem.SanitizeMol(fragment)
        rdmolops.AssignStereochemistry(fragment)
        status = rdDistGeom.EmbedMolecule(fragment)
        if status == -1:
            status = AllChem.EmbedMolecule(fragment, useRandomCoords=True)
            if status == -1:
                print('Could not embed fragment')
                sys.exit('Error: could not embed molecule')
        AllChem.MMFFOptimizeMolecule(fragment)
        for i, atom in enumerate(fragment.GetAtoms()):
            atom_id = atom.GetAtomMapNum()-1
            coord = fragment.GetConformers()[0].GetAtomPosition(i)
            coord = np.array([coord.x+0.5*n_frag*n_atoms, coord.y, coord.z])
            coordinates[atom_id, :] = coord

    rdDistGeom.EmbedMolecule(mol)
    conf = mol.GetConformer()
    for i in range(n_atoms):
        x, y, z = coordinates[i, :]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    return mol


def get_barrier_estimate(reaction_index, reactant_smiles, product_smiles,
                         structure_database=False):
    """
    Get barrier estimate based on rmsd-pp
    """
    os.mkdir(str(reaction_index))
    os.chdir(str(reaction_index))

    # write to .xyz files
    reactant_file = 'reactant.xyz'
    product_file = 'product.xyz'

    if not structure_database:
        r_mol = embed_smiles_far(reactant_smiles)
        r_mol = write_xyz_file(r_mol, reactant_file)
        p_mol = embed_smiles_far(product_smiles)
        p_mol = write_xyz_file(p_mol, product_file)
    else:
        r_mol = Chem.MolFromSmiles(reactant_smiles, sanitize=False)
        r_mol = reorder_atoms_to_map(r_mol)
        shutil.copy('../reactant.xyz', '.')
        shutil.copy('../product.xyz', '.')

    n_atoms = r_mol.GetNumAtoms()
    charge = Chem.GetFormalCharge(r_mol)

    bond_pairs = bonds_getting_formed_or_broken(reactant_smiles,
                                                product_smiles, n_atoms)

    max_xtb_energies, interpolation_points = do_rmsd_pp_calc(reactant_file, product_file, n_atoms,
                                                             reaction_index,
                                                             bond_pairs,
                                                             charge=charge)
    print(max_xtb_energies)
    n_steps = len(max_xtb_energies)

    my_index = pd.MultiIndex(levels = [[],[]], codes=[[],[]],
                             names=[u'labels', u'path_idx'])
    dataframe = pd.DataFrame(index=my_index)
    for path_idx in range(n_steps):
        #dataframe.at[reaction_index, "xTB max [Hartree]"] = max(max_xtb_energies)
        dataframe.loc[(reaction_index, path_idx), "N_steps"] = n_steps
        dataframe.loc[(reaction_index, path_idx), "reactant_smiles_am"] = reactant_smiles
        dataframe.loc[(reaction_index, path_idx), "product_smiles_am"] = product_smiles
        dataframe.loc[(reaction_index, path_idx), "maxE"] = max_xtb_energies[path_idx][0]
        dataframe.loc[(reaction_index, path_idx), "maxAB"] = max_xtb_energies[path_idx][1]
        dataframe.loc[(reaction_index, path_idx), "maxBC"] = max_xtb_energies[path_idx][2]
        #dataframe.loc[(reaction_index, path_idx), ["maxE", "maxAB", "maxBC"]] = max_xtb_energies[path_idx]
        dataframe.loc[(reaction_index, path_idx), "pointE"] = interpolation_points[path_idx][0]
        dataframe.loc[(reaction_index, path_idx), "pointAB"] = interpolation_points[path_idx][1]
        dataframe.loc[(reaction_index, path_idx), "pointBC"] = interpolation_points[path_idx][2]
        #dataframe.loc[(reaction_index, path_idx),
        #              ["pointE", "pointAB", "pointBC"]] = interpolation_points[path_idx]
    print(dataframe)
    print(reaction_index, max(max_xtb_energies), len(max_xtb_energies))
    dataframe.to_pickle(str(reaction_index)+'.pkl')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(usage='%(prog)s [options] r_smiles p_smiles r_idx')
    parser.add_argument('reactant_smiles', metavar='reactant_smiles', type=str)
    parser.add_argument('product_smiles', metavar='product_smiles', type=str)
    parser.add_argument('r_idx', metavar='r_idx', type=str)
    parser.add_argument('--use-structures',
                        action="store_true",
                        help="using structure database")

    args = parser.parse_args()
    R_SMILES = args.reactant_smiles
    P_SMILES = args.product_smiles
    REACTION_INDEX = args.r_idx
    STRUCTURE_DATABASE = args.use_structures
    #import rdkit
    #print(rdkit.__version__)
    #R_SMILES = sys.argv[1]
    #P_SMILES = sys.argv[2]
    #DATAFRAME = sys.argv[1]
    #REACTION_INDEX = sys.argv[3]
    
    get_barrier_estimate(int(REACTION_INDEX), R_SMILES, P_SMILES,
                         structure_database=STRUCTURE_DATABASE)
