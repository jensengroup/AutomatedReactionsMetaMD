import sys
import os
import numpy as np
import pandas as pd


def get_single_step_df(smiles, df):
    """
    extracts the reactions in df correspondin to single-step reactions from the
    molecule indicated in smiles
    """
    print(smiles)
    df[['reactant_smiles_am', 'product_smiles_am']] = \
        df.Reactions.str.split(pat='>>', expand=True)
    df_reverse = df[df.product_smiles_am == smiles]
    df_forward = df[df.reactant_smiles_am == smiles]

    print(df_forward)
    df_reverse = df_reverse.sort_values(by=['counts'], ascending=False)
    df_forward = df_forward.sort_values(by=['counts'], ascending=False)

    df_reverse = df_reverse.reset_index(drop=True)
    df_forward = df_forward.reset_index(drop=True)

    i = len(df_forward)
    for idx, product in zip(df_reverse.index, df_reverse.reactant_smiles_am):
        if product not in df_forward.product_smiles_am.to_list():
            df_forward.loc[i, 'reactant_smiles_am'] = smiles
            df_forward.loc[i, 'product_smiles_am'] = product
            df_forward.loc[i, 'Reactions'] = smiles+'>>'+product
            df_forward.loc[i, 'Products_canonical'] = \
                df_reverse.loc[idx, 'Reactants_canonical']
            df_forward.loc[i, 'Reactants_canonical'] = \
                df_reverse.loc[idx, 'Products_canonical']
            df_forward.loc[i, 'counts'] = 0
            i += 1
    return df_forward


def combined_run_dfs(smiles_idx, smiles, n_runs):
    """
    collects the dataframes for the n_runs meta-dynamics searches to a frame
    summarizing all reactions and one for only one-step reactions.
    """

    lowest_steps = []
    lowest_steps_above100 = []
    os.chdir(str(smiles_idx))
    combined_df = pd.DataFrame()
    for i in range(n_runs):
        try:
            df = pd.read_pickle('run'+str(i)+'.pkl')
            first_step = min(df.N_steps)
            try:
                first_step_filtered = min(df[df.N_steps > 100].N_steps)
            except ValueError:
                first_step_filtered = first_step
            combined_df = pd.concat([combined_df, df])
        except (FileNotFoundError, ValueError) as error:
            print(smiles_idx, "run crashed")
            first_step = np.nan
            first_step_filtered = np.nan

        lowest_steps.append(first_step)
        lowest_steps_above100.append(first_step_filtered)

        print(first_step, first_step_filtered)

    if len(combined_df) != 0:
        combined_df_final = \
            combined_df.groupby(['Reactions', 'Reactants_canonical',
                                 'Products_canonical']).size().reset_index(name='counts')
        print(combined_df_final.sort_values(by=['counts'], ascending=False))
        df_forward_1step = get_single_step_df(smiles, combined_df_final)

    else:
        combined_df_final = combined_df
        df_forward_1step = combined_df_final


    combined_df_final.to_csv(str(smiles_idx)+'_combined.csv')
    df_forward_1step.to_csv(str(smiles_idx)+'_1step.csv')
    print(df_forward_1step)

    os.chdir('../')
    return np.nanmean(lowest_steps), np.nanmedian(lowest_steps), np.nanmean(lowest_steps_above100), np.nanmedian(lowest_steps_above100)


if __name__ == '__main__':
    OUT_DF = pd.DataFrame()
    N_RUNS = 100
    IN_SMILES_DF = pd.read_csv(sys.argv[1], index_col=0)
    print(IN_SMILES_DF)
    for IDX, SMILES in zip(IN_SMILES_DF.index, IN_SMILES_DF.smiles):
        output = combined_run_dfs(IDX, SMILES, N_RUNS)
        OUT_DF.loc[IDX, 'smiles'] = SMILES
        OUT_DF.loc[IDX, 'mean_step'] = output[0]
        OUT_DF.loc[IDX, 'median_step'] = output[1]
        OUT_DF.loc[IDX, 'mean_step_above100'] = output[2]
        OUT_DF.loc[IDX, 'median_step_above100'] = output[3]

    print(OUT_DF)
    #OUT_DF.to_csv("steps.csv")

