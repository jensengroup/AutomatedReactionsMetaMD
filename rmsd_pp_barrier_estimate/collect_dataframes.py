import pandas as pd
import sys


job_df = pd.read_csv(sys.argv[1])

my_index = pd.MultiIndex(levels = [[],[]], codes=[[],[]], names=[u'labels', u'path_idx'])

df_upd = pd.DataFrame(index=my_index)
#print(df)
for idx in job_df.index:
    print(idx)
    reactant_idx = job_df.loc[idx, 'reactant']
    r_idx = job_df.loc[idx, 'r_idx']
    letter = job_df.loc[idx, 'letter']
    #print(reactant_idx, r_idx, letter)
    pkl_file = str(reactant_idx)+'/'+str(reactant_idx)+'_'+str(r_idx)+'_'+letter+'.pkl'
    df = pd.read_pickle(pkl_file)
    if letter=='d' or letter=='e':
        print('now')
        print(df)
        #df=df.reindex(columns=['N_steps', 'product_smiles_am', 'reactant_smiles_am', 'maxE', 'maxAB', 'maxBC', 'pointE', 'pointAB', 'pointBC'])
        df_map = {'reactant_smiles_am':'product_smiles_am', 'product_smiles_am':'reactant_smiles_am'}
        df.rename(columns={**df_map, **{v:k for k,v in df_map.items()}}, inplace=True)
        print(df)
    df_upd = pd.concat([df_upd, df])


print(df_upd)
df_upd.to_csv("job_df_upd.csv")

