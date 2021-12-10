import sys
import os
import tarfile
import pandas as pd


def extract_products(rxn_idx, file_name="initial_structures.xyz", n_runs=100):
    os.chdir(rxn_idx)
    rxn_df = pd.read_csv(str(rxn_idx)+'_1step.csv', index_col=0)
    rxn_df = rxn_df[rxn_df.counts != 0]
    products = set(rxn_df.product_smiles_am)
    run = 0
    while len(products) != 0 and run < n_runs:
        print(run, len(products))
        if os.path.exists('run'+str(run)+'.pkl'):
            run_df = pd.read_pickle('run'+str(run)+'.pkl')
        else:
            run += 1
            continue
        for reaction in run_df.Reactions:
            if reaction.split('>>')[1] in products and reaction in rxn_df.Reactions.to_list():
                idx = rxn_df.index[rxn_df.Reactions == reaction].to_list()[0]
                products.remove(reaction.split('>>')[1])

                tar = tarfile.open('run'+str(run)+'_database.tar.gz', "r:gz")
                structure_df = tar.extractfile('run'+str(run)+'/structure_database/structure_database.csv')
                structure_df = pd.read_csv(structure_df, index_col=0)

                p_hash = structure_df.index[structure_df.smiles == reaction.split('>>')[1]].to_list()[0]
                p_file_name = 'run'+str(run)+'/structure_database/xyz_files/'+str(p_hash)+'.xyz'
                print(p_file_name)
                p_file = tar.extractfile(p_file_name)
                contents = p_file.read()

                with open(file_name, "a+b") as s_file:
                    #contents = "".join(contents)
                    s_file.write(contents)
        run += 1





    os.chdir('../')



if __name__ == "__main__":
    RXN_IDX = sys.argv[1]
    extract_products(RXN_IDX)


