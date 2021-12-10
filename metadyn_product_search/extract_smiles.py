#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python
"""
get mapped smiles from structure and check charges
"""

import sys

from reaction_box_no_slurm import get_smiles


xyz_file = sys.argv[1]
charge = int(sys.argv[2])
smiles, formal_charge, _ = get_smiles(xyz_file, charge, check_ac=False)
print(smiles, formal_charge)
print(charge == formal_charge)
