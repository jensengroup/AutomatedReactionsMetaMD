#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python


import os
import textwrap
import sys
import time
import random

import pandas as pd

from collect_products import extract_products


def qsub_prep(script_path, cpus, mem, smiles_idx, run_nr, smiles, s_factor,
              time_ps, k_push, alp, random_seed, with_products=False):
    """
    write qsub file for SLURM subsmissin
    """
    pwd = os.getcwd()

    qsub_file = """\
    #!/bin/sh
    #SBATCH --job-name={3}_{4}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task={1}
    #SBATCH --mem={2}
    #SBATCH --error={10}/{3}/run{4}.stderr
    #SBATCH --output={10}/{3}/run{4}.stdout
    #SBATCH --ntasks=1
    #SBATCH --time=8:00:00
    #SBATCH --partition=kemi1
    #SBATCH --no-requeue

    #mkdir /scratch/$SLURM_JOB_ID
    #cp ../initial_structures.xyz /scratch/$SLURM_JOB_ID
    cd /scratch/$SLURM_JOB_ID

    #run python code

    ({0} {3} {4} '{5}' {6} {7} {8} {9} {11} {12})
    #cp output back

    cp run{4}/dataframe.pkl {10}/{3}/run{4}.pkl
    tar -zcvf structure_database.tar.gz run{4}/structure_database
    tar -zcvf run{4}.tar.gz run{4}

    cp run{4}.tar.gz {10}/{3}/run{4}.tar.gz
    cp structure_database.tar.gz {10}/{3}/run{4}_database.tar.gz

    rm {10}/{3}_run{4}_qsub.tmp

    #rm -r /scratch/$SLURM_JOB_ID

    """.format(script_path, cpus, mem, smiles_idx, run_nr, smiles, s_factor,
               time_ps, k_push, alp, pwd, random_seed, with_products)

    with open(str(smiles_idx) + '_run'+str(run_nr) + "_qsub.tmp", "w") as qsub:
        qsub.write(textwrap.dedent(qsub_file))

    return str(smiles_idx) + '_run' + str(run_nr) + "_qsub.tmp"


def run_calculations(smiles_df, n_runs, script_path, s_factor, time_ps, k_push,
                     alp, max_queue, cpus, mem, random_seeds,
                     with_products=False):
    """
    For each given smiles - submit n_runs metadynamics searches with the given
    parameters. Only submit new jobs when less than max_queue jobs in the queue
    """
    submitted_jobs = set()
    submitted_smiles = set()
    for j, idx in enumerate(smiles_df.index):
        smiles = smiles_df.loc[idx, 'smiles']
        if idx not in submitted_smiles:
            os.mkdir(str(idx))
            submitted_smiles.add(idx)
        for run_nr in range(n_runs):
            qsub_name = qsub_prep(script_path, cpus, mem, idx, run_nr, smiles,
                                  s_factor, time_ps, k_push, alp,
                                  random_seeds[j*n_runs+run_nr],
                                  with_products=with_products)
            if with_products:
                with open(qsub_name, 'r') as qsub:
                    contents = qsub.readlines()
                contents.insert(13, "cp {}_initial_structures.xyz /scratch/$SLURM_JOB_ID\n".format(idx))
                with open(qsub_name, "w") as qsub:
                    contents = "".join(contents)
                    qsub.write(contents)
            slurmid = os.popen("sbatch " + qsub_name).read()
            slurmid = int(slurmid.strip().split()[-1])

            submitted_jobs.add(slurmid)

            if len(submitted_jobs) >= max_queue:
                while True:
                    job_info = os.popen("squeue -u mharris").readlines()[1:]
                    current_jobs = {int(job.split()[0]) for job in job_info}

                    if len(current_jobs) >= max_queue:
                        time.sleep(30)
                    else:
                        finished_jobs = submitted_jobs - current_jobs
                        print("finished jobs: ", finished_jobs)
                        for job in finished_jobs:
                            submitted_jobs.remove(job)
                        break


if __name__ == "__main__":
    SMILES_LIST = sys.argv[1]
    SMILES_DF = pd.read_csv(SMILES_LIST, index_col=0)
    # print(SMILES_DF)
    INITIALIZE_PRODUCTS = False
    CPUS = 1
    MEM = "2GB"
    MAX_QUEUE = 300
    N_RUNS = 100
    S_FACTOR = 0.8
    TIME_PS = 5
    K_PUSH = 0.05
    ALP = 0.3

    SCRIPT = '/groups/kemi/mharris/github/metadyn_product_search/reaction_box_no_slurm.py'

    _DIR = str(S_FACTOR)+'_'+str(K_PUSH)+'_'+str(ALP)

    if INITIALIZE_PRODUCTS:
        os.mkdir(_DIR+'_products')
        os.chdir(_DIR)
        for idx in SMILES_DF.index:
            extract_products(str(idx), file_name='../../'+_DIR+'_products/'+str(idx)+"_initial_structures.xyz")
        os.chdir("../"+_DIR+'_products')
    else:
        os.mkdir(_DIR)
        os.chdir(_DIR)

    # shutil.copy('../1_initial_structures.xyz', '.')
    RANDOM_SEEDS = [random.randint(1, 100000) for _ in range(len(SMILES_DF)*N_RUNS)]
    run_calculations(SMILES_DF, N_RUNS, SCRIPT, S_FACTOR, TIME_PS, K_PUSH,
                     ALP, MAX_QUEUE, CPUS, MEM, RANDOM_SEEDS,
                     with_products=INITIALIZE_PRODUCTS)

