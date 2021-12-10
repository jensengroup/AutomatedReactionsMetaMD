#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python


import sys
from run_dft_ts import bonds_getting_formed_or_broken, check_ts, get_smiles, atom_information


if __name__ == "__main__":
    r_file = sys.argv[1]
    p_file = sys.argv[2]
    ts_file = sys.argv[3]
    cpus = sys.argv[4]
    mem = sys.argv[5]
    charge = 0
    print(r_file, cpus)

    rsmi, _, _ = get_smiles(r_file, charge)
    psmi, _, _ = get_smiles(p_file, charge)
    print(rsmi, psmi)
    n_atoms, atom_numbers = atom_information(ts_file)
    bond_pairs = bonds_getting_formed_or_broken(rsmi, psmi, n_atoms)
    ts_found = check_ts(ts_file, rsmi, psmi, n_atoms, atom_numbers, bond_pairs,
                        cpus, mem)

    if not ts_found:
        print("---------- do constrained optimization ------------")
