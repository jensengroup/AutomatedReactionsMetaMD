import sys
import os
import math
import tarfile
import numpy as np
import pandas as pd

def create_job_df(r_idx, structure_database=True):
    print(r_idx)
    count_lists = []
    os.chdir(r_idx)
    rxn_df = pd.read_csv(str(r_idx)+'_1step.csv', index_col=0)
    if rxn_df.empty:
        os.chdir("../")
        print("no reactions found for reactant {}".format(r_idx))
        return
    rxn_df = rxn_df[rxn_df.counts != 0]
    rxn_df['count_list'] = None
    #print(rxn_df)
    for j, count in enumerate(rxn_df.counts):
        #print('count:', count)
        if count >= 5:
            count_list =[1]*5
        else:
            count_list = []
            i = math.ceil(5/count)
            count_list.append(math.ceil(5/count))
            while i < 5:
                if i+math.ceil(5/count) > 5:
                    count_list.append(5-i)
                    i=5
                else:
                    count_list.append(math.ceil(5/count))
                    i+=math.ceil(5/count)
        count_lists.append(count_list)
        rxn_df.at[j, 'count_list'] = np.array(count_list)
    #rxn_df.loc[:, 'count_list'] = np.array(count_lists, dtype=object)
    rxn_df.loc[:, 'current_index'] = [0]*len(rxn_df)
    rxn_df.loc[:, 'current_letter'] = ['a']*len(rxn_df)
    #print(rxn_df)
    global JOB_DF
    #mult_index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['r_idx', 'letter'])
    #job_df = pd.DataFrame(index=mult_index)
    for run in range(100):
        if os.path.exists('run'+str(run)+'.pkl'):
            run_df = pd.read_pickle('run'+str(run)+'.pkl')
        else:
            continue
        for reaction in run_df.Reactions:
            if reaction in rxn_df.Reactions.to_list() or reaction.split('>>')[1]+'>>'+reaction.split('>>')[0] in rxn_df.Reactions.to_list():
                idx = rxn_df.index[rxn_df.Reactions == reaction].to_list()
                forward = True
                if len(idx) > 1:
                    print("somethings wrong: reaction has 2 matches")
                if not idx:
                    idx = rxn_df.index[rxn_df.Reactions == reaction.split('>>')[1]+'>>'+reaction.split('>>')[0]].to_list()
                    forward = False
                #print(idx)
                if not idx:
                    print(reaction)
                    print(rxn_df.Reactions.to_list())
                idx = idx[0]
                cur_idx = rxn_df.loc[idx, "current_index"]
                count_list = rxn_df.loc[idx, 'count_list']
                if (cur_idx + 1) > len(count_list):
                    continue

                tar = tarfile.open('run'+str(run)+'_database.tar.gz', "r:gz")
                structure_df = tar.extractfile('run'+str(run)+'/structure_database/structure_database.csv')
                structure_df = pd.read_csv(structure_df, index_col=0)
                i1 = 0
                i2 = 1
                if not forward:
                    i1, i2 = i2, i1

                r_hash = structure_df.index[structure_df.smiles == reaction.split('>>')[i1]].to_list()[0]
                p_hash = structure_df.index[structure_df.smiles == reaction.split('>>')[i2]].to_list()[0]
                r_file_name = 'run'+str(run)+'/structure_database/xyz_files/'+str(r_hash)+'.xyz'
                p_file_name = 'run'+str(run)+'/structure_database/xyz_files/'+str(p_hash)+'.xyz'
                for _ in range(count_list[cur_idx]):
                    current_letter = rxn_df.loc[idx, 'current_letter']
                    JOB_DF.loc[(r_idx, idx, current_letter), 'tarfile'] = 'run'+str(run)+'_database.tar.gz'
                    JOB_DF.loc[(r_idx, idx, current_letter), 'reactant_smiles_am'] = reaction.split('>>')[i1]
                    JOB_DF.loc[(r_idx, idx, current_letter), 'product_smiles_am'] = reaction.split('>>')[i2]
                    JOB_DF.loc[(r_idx, idx, current_letter), 'r_path'] = r_file_name
                    JOB_DF.loc[(r_idx, idx, current_letter), 'p_path'] = p_file_name
                    current_letter = chr(ord(current_letter) + 1)
                    #print(current_letter)
                    rxn_df.loc[idx, 'current_letter'] = current_letter
                rxn_df.loc[idx, 'current_index'] += 1


    os.chdir('../')




if __name__ == "__main__":
    SMILES_DF = pd.read_csv(sys.argv[1], index_col=0)
    MULT_INDEX = pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]], names=['reactant', 'r_idx', 'letter'])
    JOB_DF =pd.DataFrame(index=MULT_INDEX)
    for i in SMILES_DF.index:
        create_job_df(str(i))

    JOB_DF.to_csv('job_df_test.csv')

