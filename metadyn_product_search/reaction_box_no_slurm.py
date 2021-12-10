#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python

"""
Find reactant products for the input SMILES (atom-mapped) string using
xTB meta-dynamics
"""

import os
import sys
import shutil
import subprocess
import textwrap
import hashlib
import random
import xyz2mol_local
import numpy as np
import pandas as pd
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Chem import rdmolops

from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem


def run_cmd(cmd):
    """
    Run command line
    """
    cmd = cmd.split()
    print(cmd)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    return output.decode('utf-8')


def set_up_directory(smiles_idx):
    """
    checks if directory for SMILES is set up, otherwise creates it
    """
    if os.path.exists(str(smiles_idx)):
        return

    os.mkdir(str(smiles_idx))
    os.mkdir(str(smiles_idx)+'/md')
    os.mkdir(str(smiles_idx)+'/metadynamics')

    shutil.copy('md.inp', str(smiles_idx)+'/md')
    shutil.copy('metadyn.inp', str(smiles_idx))


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


def extract_last_structure(trj_file, last_structure_name):
    """
    Extracts the last structure in a trajectory file
    """

    with open(trj_file, 'r') as _file:
        line = _file.readline()
        n_lines = int(line.split()[0])+2
    count = 0
    input_file = open(trj_file, 'r')
    dest = None
    for line in input_file:
        if count % n_lines == 0:
            if dest:
                dest.close()
            dest = open(last_structure_name, "w")
        count += 1
        dest.write(line)

def embed_smiles_far(smiles):
    """
    create 3D conformer with atom order matching atom mapping. If more than one
    fragment present, they are moved 0.5*n_atoms from each other
    """

    embedded_fragments = []
    mol_sizes = []

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = reorder_atoms_to_map(mol)

    fragments = Chem.GetMolFrags(mol, asMols=True)
    n_atoms = mol.GetNumAtoms()
    coordinates = np.zeros((n_atoms, 3))


    for fragment in fragments:
        Chem.SanitizeMol(fragment)
        rdmolops.AssignStereochemistry(fragment)
        status = AllChem.EmbedMolecule(fragment, maxAttempts=10000,
                                       randomSeed=RANDOM_SEED)
        if status == -1:
            print('fragment could not be embedded')
            sys.exit("Error: could not embed molecule")

        AllChem.MMFFOptimizeMolecule(fragment)
        embedded_fragments.append(fragment)
        dm = AllChem.Get3DDistanceMatrix(fragment)
        biggest_distance = np.amax(dm)
        mol_sizes.append(biggest_distance)

    np.random.seed(seed=RANDOM_SEED)


    for n_frag, fragment in enumerate(embedded_fragments):
        if n_frag == 0:
            random_vector = np.zeros(3)
        else:
            random_vector = np.random.rand(3)*2-1
            random_vector = random_vector / np.linalg.norm(random_vector)

        conformer = fragment.GetConformer()
        translation_distance = 0.5*(mol_sizes[n_frag]+mol_sizes[0])+2

        for i, atom in enumerate(fragment.GetAtoms()):
            atom_id = atom.GetAtomMapNum()-1
            coord =  conformer.GetAtomPosition(i)
            coord += Point3D(*(translation_distance*random_vector))
            coordinates[atom_id, :] = coord

    rdDistGeom.EmbedMolecule(mol)
    conf = mol.GetConformer()
    for i in range(n_atoms):
        x, y, z = coordinates[i, :]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    return mol



def write_xyz_file(mol, file_name, smiles):

    """
    Embeds a mol object to get 3D coordinates which are written to an .xyz file
    """

    n_atoms = mol.GetNumAtoms()
    charge = Chem.GetFormalCharge(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    Chem.SanitizeMol(mol)
    rdmolops.AssignStereochemistry(mol)
    #AllChem.EmbedMolecule(mol, maxAttempts=10000, randomSeed=RANDOM_SEED)
    #status = AllChem.MMFFOptimizeMolecule(mol, ignoreInterfragInteractions=False)
    #print(status)
    #if status == -1:
    mol = embed_smiles_far(smiles)

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


def write_md_input(scale_factor, time=2):
    """
    Write the input file for an MD xTB calculation based on the input
    variables
    """
    md_file = """\
    $md
       time={1}
       step=0.4
       temp=300
       hmass=2
    $end
    $scc
       temp=300
    $end
    $cma
    $wall
       potential=logfermi
       beta=10.0
       temp=6000
       sphere: auto, all
       autoscale={0}
    $end
    """.format(scale_factor, time)

    with open('md.inp', 'w') as ofile:
        ofile.write(textwrap.dedent(md_file))

def write_meta_input(time, kpush, alp, scale_factor, structure_file=None):

    """
    Write the input file for an meta-dynamics xTB calculation based on the input
    variables
    """
    meta_file = """\
    $md
       time={0}
       step=0.4
       temp=300
       hmass=2
       shake=0
       dump=10
    $end
    $metadyn
       save=100
       kpush={1}
       alp={2}
    $end
    $scc
       temp=6000
    $end
    $cma
    $wall
       potential=logfermi
       beta=10.0
       temp=6000
       sphere: auto, all
       autoscale={3}
    $end
    """.format(time, kpush, alp, scale_factor)

    with open('metadyn.inp', 'w') as ofile:
        ofile.write(textwrap.dedent(meta_file))

    if structure_file:
        with open('metadyn.inp', 'r') as f:
            contents = f.readlines()
        contents.insert(9, "   coord={}\n".format(structure_file))
        with open('metadyn.inp', 'w') as f:
            contents = "".join(contents)
            f.write(contents)

def calculate_md_relaxed_structure(smiles, scale_factor, ridx):
    """
    This function submits an md with a box with size scaled by scale_factor and
    extracts last structure of the trajectory file
    """

    os.mkdir('md')
    os.chdir('md')
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = reorder_atoms_to_map(mol)
    n_atoms = mol.GetNumAtoms()
    charge = GetFormalCharge(mol)
    write_xyz_file(mol, str(ridx)+'.xyz', smiles)
    write_md_input(scale_factor)
    output = run_cmd("/groups/kemi/koerstz/opt/xtb/6.1/bin/xtb {0} --omd --input md.inp --gfn2 --chrg {1}".format(str(ridx)+'.xyz', charge))

    with open('md_out.log', 'w') as _file:
        _file.write(output)

    out_file = str(scale_factor)+'_md.xyz'
    extract_last_structure('xtb.trj', out_file)
    
    check_md_reaction(out_file, charge, smiles, str(ridx)+'.xyz')
    shutil.copy(out_file, '../')

    os.chdir('../')
    return out_file, charge, n_atoms


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



def extract_smiles(xyz_file, charge, allow_charge=True, check_ac=False):
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

    if check_ac:
        global AC_SAME
        ac = Chem.GetAdjacencyMatrix(input_mol)
        if not np.all(AC == ac):
            AC_SAME = False
            print("change in AC: stopping")

    return structure_smiles, GetFormalCharge(structure_mol), res_status


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

def get_smiles(xyz_file, charge, check_ac=False):
    """
    Try different things to extract sensible smiles using xyz2mol
    """
    smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                       allow_charge=True,
                                                       check_ac=check_ac)
    if formal_charge != charge:
        smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                           allow_charge=False,
                                                           check_ac=check_ac)
    #except:
    #    try:
    #        smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
    #                                                           allow_charge=False,
    #                                                           check_ac=check_ac)
    #    except:
    #        return None, None, None

    return smiles, formal_charge, res_status

def check_md_reaction(md_xyz, charge, in_smiles, in_xyz):
    """
    Checks if a reaction allready occurred in the MD run: indicating
    barrierless or very low barrier
    """

    smiles, formal_charge, res_status1 = get_smiles(md_xyz, charge,
                                                   check_ac=True)
    if not smiles:
        print("something's fishy with the MD structures")
        sys.exit()

    if formal_charge != charge:
        print("something's fishy with the MD structures charges")
        sys.exit()


    if smiles != in_smiles:
        print("smiles changed during MD")
        global n_steps_list_opt
        global reactions
        global canonical_reactants
        global canonical_products
        global smiles_list
        global MD_REACTION

        MD_REACTION=True
        n_steps_list_opt = []
        reactions = []
        canonical_reactants = []
        canonical_products = []
        smiles_list = []


        os.mkdir("../structure_database")
        os.mkdir("../structure_database/xyz_files")

        pd.DataFrame().to_csv("../structure_database/structure_database.csv")

        database_dir = os.path.join('../', 'structure_database')

        r_canonical = canonicalize_smiles(in_smiles)
        canonical_reactants.append(r_canonical)

        optsmiles, formal_charge, res_status2 = get_smiles("xtbopt.xyz", charge,
                                                          check_ac=True)
        if optsmiles != in_smiles:
            print("warning: smiles changes during pre-MD optimization!!!")
            p_canonical = canonicalize_smiles(optsmiles)
            canonical_products.append(p_canonical)
            reactions.append(in_smiles+'>>'+optsmiles)
            save_structure_to_database(optsmiles, "xtbopt.xyz", res_status2,
                                       database_dir)
            smiles_list.append(in_smiles)
            smiles_list.append(optsmiles)


        else:
            reactions.append(in_smiles+'>>'+smiles)

            p_canonical = canonicalize_smiles(smiles)

            canonical_products.append(p_canonical)
            save_structure_to_database(smiles, md_xyz, res_status1, database_dir)

            smiles_list.append(in_smiles)
            smiles_list.append(smiles)

        save_structure_to_database(in_smiles, in_xyz, np.nan, database_dir)

        n_steps_list_opt.append(0)

        if AC_SAME == False:
            global DF
            DF['Reactions'] = reactions
            DF['Reactants_canonical'] = canonical_reactants
            DF['Products_canonical'] = canonical_products
            DF['N_steps'] = n_steps_list_opt

            DF.to_pickle('../dataframe.pkl')





def split_trajectory(trajectory_file, charge, current_time, n_check=10):
    """
    Check trajectory for reactions every n_check points
    """
    _dir = 'analyze_trajectory_'+str(current_time)
    os.mkdir(_dir)
    smiles_list = []
    n_steps_list = []
    count = 0
    saved_struc = 1
    global n_steps
    if not RESTARTED:
        n_steps = 0
    with open(trajectory_file, 'r') as _file:
        line = _file.readline()
        n_lines = int(line.split()[0])+2
        while line:
            if count % (n_lines*n_check) == 0:
                print(count, saved_struc)
                file_name = _dir+'/'+str(saved_struc)+'.xyz'
                with open(file_name, 'w') as xyz_file:
                    for _ in range(n_lines):
                        xyz_file.write(line)
                        line = _file.readline()
                        count += 1
                try:
                    smiles, formal_charge, _ = extract_smiles(file_name,
                                                              charge, 
                                                              allow_charge=True, 
                                                              check_ac=False)
                    print(smiles)
                    if formal_charge != charge:
                        print(formal_charge, charge)
                        smiles, formal_charge, _ = extract_smiles(file_name,
                                                                  charge, 
                                                                  allow_charge=False,
                                                                  check_ac=False)
                        if formal_charge != charge:
                            continue
                    if not smiles_list and formal_charge == charge:
                        smiles_list.append(smiles)
                        print('smiles =', smiles)
                        saved_struc += 1
                        n_steps_list.append(n_steps)
                    elif smiles != smiles_list[-1] and formal_charge == charge:
                        smiles_list.append(smiles)
                        print('smiles = ', smiles)
                        saved_struc += 1
                        n_steps_list.append(n_steps)
                except:
                    print("error reading smiles")
                    pass
                n_steps += n_check
            else:
                line = _file.readline()
                count += 1

    n_steps_list.append(n_steps)

    return n_steps_list

def save_structure_to_database(smiles, xyzfile, res_status, database_dir):
    """
    The saved SMILES is represented by its SHA1 code. Then the database of
    optimized structures is checked for that hash code. If not present in the
    database: save the xyzfile with the hash name and add entrance to the .csv
    file for the structure database
    """
    #database_dir = os.path.join('../../', 'structure_database')

    hashobject = hashlib.sha1(smiles.encode())
    hashcode = hashobject.hexdigest()

    database_path = os.path.join(database_dir, 'structure_database.csv')
    database = pd.read_csv(database_path, index_col=0)

    if hashcode not in database.index:
        database.loc[hashcode, 'smiles'] = smiles
        database.loc[hashcode, 'resonance_run'] = res_status
        structure_path = os.path.join(database_dir, 'xyz_files',
                                      str(hashcode)+'.xyz')
        shutil.copy(xyzfile, structure_path)

    database.to_csv(database_path)



def check_opt_xyz(n_steps_list, charge):
    """
    goes through the initially saved structures to see if they still (after
    optimization) correspond to a reaction haven taken place.
    """
    global n_steps_list_opt
    global reactions
    global canonical_reactants
    global canonical_products
    global smiles_list

    opt_files = [f for f in os.listdir(os.curdir) if f.endswith("optxyz")]
    opt_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    os.mkdir("optimized_structures")
    if not RESTARTED and not MD_REACTION:
        n_steps_list_opt = []
        reactions = []
        canonical_reactants = []
        canonical_products = []
        smiles_list = []

        os.mkdir("../../structure_database")
        os.mkdir("../../structure_database/xyz_files")
        pd.DataFrame().to_csv("../../structure_database/structure_database.csv")

    database_dir = os.path.join('../../', 'structure_database')

    for i, optfile in enumerate(opt_files):
        try:
            smiles, formal_charge, res_status = extract_smiles(optfile, charge,
                                                               allow_charge=True,
                                                               check_ac=True)
            if formal_charge != charge:
                smiles, formal_charge, res_status = extract_smiles(optfile, charge,
                                                                   allow_charge=False,
                                                                   check_ac=True)
        except:
            try:
                smiles, formal_charge, res_status = extract_smiles(optfile, charge,
                                                                   allow_charge=False,
                                                                   check_ac=True)
            except:
                continue

        if formal_charge != charge:
            continue

        if not smiles_list:
            smiles_list.append(smiles)
            shutil.copy(optfile, 'optimized_structures')
            save_structure_to_database(smiles, optfile, res_status,
                                       database_dir)

        elif smiles != smiles_list[-1]:
            reactant = smiles_list[-1]
            reaction = reactant+'>>'+smiles
            reactions.append(reaction)

            canonical_reactant = canonicalize_smiles(smiles_list[-1])
            canonical_reactants.append(canonical_reactant)
            canonical_product = canonicalize_smiles(smiles)
            canonical_products.append(canonical_product)

            smiles_list.append(smiles)
            n_steps_list_opt.append(n_steps_list[i])
            shutil.copy(optfile, 'optimized_structures')
            save_structure_to_database(smiles, optfile, res_status,
                                       database_dir)

    os.chdir('../')
    return


def setup_restart(time_ps, n_atoms, current_time, structure_file, s=0.8,
                  with_products=False, multi_fragment=False):
    """
    goes through the structures saved every ps and appends to coordinate file
    ready for a restarted meta-dynamics simulation
    """
    b_to_A = 0.529177249
    if multi_fragment:
        new_s = s-0.02
    else: 
        new_s = s

    for i in range(int(time_ps)-1):
        coords = pd.read_csv("scoord."+str(i+1), delim_whitespace=True, skiprows=1,
                             nrows=n_atoms, header=None)[[3, 0, 1, 2]]

        coords.iloc[:, 1:] = coords.iloc[:, 1:].mul(b_to_A)


        with open(structure_file, 'a') as ofile:
            ofile.write(str(n_atoms)+'\n\n')
            np.savetxt(ofile, coords.to_numpy(), delimiter=' ', fmt='%s')

    extract_last_structure("xtb.trj", "restart_structure.xyz")
    shutil.copy("xtb.trj", "xtb_"+str(current_time)+".trj")

    with open("metadyn.inp", "r") as f:
        contents = f.readlines()

    for i, line in enumerate(contents):
        if "autoscale" in line:
            new_line = line.replace(str(s), str(new_s))
            line_idx = i

    contents[i-1] = new_line


    with open("metadyn.inp", "w") as f:
        contents = "".join(contents)
        f.write(contents)


    global RESTARTED
    if not RESTARTED:

        with open("metadyn.inp", "r") as f:
            contents = f.readlines()

        contents.insert(1, "   restart=true\n")
        if not with_products:
            contents.insert(10, "   coord={}\n".format(structure_file))

        with open("metadyn.inp", "w") as f:
            contents = "".join(contents)
            f.write(contents)

        RESTARTED = True

    return new_s


def do_metadynamics_calculation(md_file, time, k_push, alp, scale_factor,
                                charge, current_time, structure_file=None):
    """
    runs a meta-dynamics calculation with the given input variables
    """
    if not RESTARTED:
        os.mkdir("metadynamics")
        shutil.copy(md_file, "metadynamics")
        if structure_file:
            shutil.copy('../'+structure_file, 'metadynamics')
        os.chdir("metadynamics")
        write_meta_input(time, k_push, alp, scale_factor,
                         structure_file=structure_file)

    output = run_cmd("/groups/kemi/koerstz/opt/xtb/6.1/bin/xtb {0} --md --input metadyn.inp --gfn2 --chrg {1}".format(md_file, charge))

    with open("meta_out.log", "w") as _file:
        _file.write(output)

    n_steps_list = split_trajectory('xtb.trj', charge, current_time)

    os.chdir('analyze_trajectory_'+str(current_time))

    xyz_files = [f for f in os.listdir() if f.endswith('.xyz')]
    for xyz_file in xyz_files:
        output = run_cmd("/groups/kemi/koerstz/opt/xtb/6.1/bin/xtb {0} --opt tight --gfn2 --chrg {1}".format(xyz_file, charge))
        os.rename('xtbopt.xyz', xyz_file[:-4]+'.optxyz')

    check_opt_xyz(n_steps_list, charge)


if __name__ == "__main__":
    os.environ["XTBHOME"] = "/groups/kemi/koerstz/opt/xtb/6.1/bin"
    os.environ["OMP_STACKSIZE"] = '2G'
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ["MKL_NUM_THREADS"] = '1'
    os.system('ulimit -s unlimited')

    RESTARTED=False
    MD_REACTION=False
    SMILES_IDX = sys.argv[1]
    RUN_NR = sys.argv[2]
    #SMILES = '[N:1](=[c:2]1\\[n:3][n:4][o:5][n:6][c:7]1[H:9])\\[H:8]'
    SMILES = sys.argv[3]
    SCALE_FACTOR = sys.argv[4]
    TIME = sys.argv[5]  #time in ps
    K_PUSH = sys.argv[6]
    ALP = sys.argv[7]
    global RANDOM_SEED
    RANDOM_SEED = int(sys.argv[8])
    WITH_PRODUCTS = sys.argv[9] == "True"
    print(WITH_PRODUCTS)
    print(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    os.mkdir('run'+str(RUN_NR))
    if WITH_PRODUCTS:
        #shutil.copy(str(SMILES_IDX)+'_initial_structures.xyz', 'run'+str(RUN_NR))
        STRUCTURE_FILE = str(SMILES_IDX)+'_initial_structures.xyz'
    else:
        STRUCTURE_FILE = None
    #STRUCTURE_FILE = str(SMILES_IDX)+'_initial_structures.xyz'
    print(STRUCTURE_FILE)
    os.chdir('run'+str(RUN_NR))
    MAX_TIME = 200

    MOL = Chem.MolFromSmiles(SMILES, sanitize=False)
    MOL = reorder_atoms_to_map(MOL)
    AC = Chem.GetAdjacencyMatrix(MOL)
    DF = pd.DataFrame()
    AC_SAME = True
    CURRENT_TIME = 0

    MULTI_FRAGMENT = not len(SMILES.split('.')) == 1

    MD_FILE, CHARGE, N_ATOMS = calculate_md_relaxed_structure(SMILES, SCALE_FACTOR,
                                                              SMILES_IDX)

    while AC_SAME and CURRENT_TIME < MAX_TIME:
        do_metadynamics_calculation(MD_FILE, TIME, K_PUSH, ALP, SCALE_FACTOR,
                                    CHARGE, CURRENT_TIME,
                                    structure_file=STRUCTURE_FILE)
        STRUCTURE_FILE = str(SMILES_IDX)+'_initial_structures.xyz'
        SCALE_FACTOR = setup_restart(TIME, N_ATOMS, CURRENT_TIME,
                                     STRUCTURE_FILE, s=np.float(SCALE_FACTOR),
                                     with_products=WITH_PRODUCTS,
                                     multi_fragment=MULTI_FRAGMENT)
        CURRENT_TIME += int(TIME)
        MD_FILE = "restart_structure.xyz"


    DF['Reactions'] = reactions
    DF['Reactants_canonical'] = canonical_reactants
    DF['Products_canonical'] = canonical_products
    DF['N_steps'] = n_steps_list_opt

    DF.to_pickle('../dataframe.pkl')


