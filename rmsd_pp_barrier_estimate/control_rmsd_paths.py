#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python

import os
import textwrap
import sys
import time

import pandas as pd


def qsub_prep_structure(job_index, reactant_idx, reaction_idx, letter,
                        r_smiles, p_smiles, script_path, cpus, mem, tarfile,
                        r_path, p_path):
    """
    Write qsub file for SLURM submission
    """
    pwd = os.getcwd()

    qsub_file = """\
    #!/bin/sh
    #SBATCH --job-name={0}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task={1}
    #SBATCH --mem={2}
    #SBATCH --error={3}/{0}.stderr
    #SBATCH --output={3}/{0}.stdout
    #SBATCH --ntasks=1
    #SBATCH --time=8:00:00
    #SBATCH --partition=kemi1
    #SBATCH --no-requeue

    tar -zxvf {10}/{4} -C /scratch/$SLURM_JOB_ID --transform='s,.*/,,' {5}
    tar -zxvf {10}/{4} -C /scratch/$SLURM_JOB_ID --transform='s,.*/,,' {6}

    cd /scratch/$SLURM_JOB_ID

    mv $(basename {5}) reactant.xyz
    mv $(basename {6}) product.xyz

    #run python code
    ({7} --use-structures '{8}' '{9}' {0})
    #cp output back

    tar -zcvf {10}_{11}_{12}.tar.gz {0}

    mv {0}.pkl {10}_{11}_{12}.pkl
    cp {10}_{11}_{12}.tar.gz {3}/{10}
    cp {10}_{11}_{12}.pkl {3}/{10}


    rm {3}/{0}_qsub.tmp

    """.format(job_index, cpus, mem, pwd, tarfile, r_path, p_path, script_path,
               r_smiles, p_smiles, reactant_idx, reaction_idx, letter)

    with open(str(job_index) + "_qsub.tmp", "w") as qsub:
        qsub.write(textwrap.dedent(qsub_file))

    return str(job_index) + "_qsub.tmp"


def run_calculations_structures(job_df, script_path, cpus, mem, max_queue):
    """
    For each reaction in job_names: calculates an rmsd-pp barrier estimate
    submits jobs when queue empty
    """
    submitted_jobs = set()
    for i in job_df.index:
        reactant_idx = job_df.loc[i, 'reactant']
        reaction_idx = job_df.loc[i, 'r_idx']
        letter = job_df.loc[i, 'letter']
        if letter in ['a', 'b', 'c']:
            r_smiles = job_df.loc[i, 'reactant_smiles_am']
            p_smiles = job_df.loc[i, 'product_smiles_am']
            r_path = job_df.loc[i, 'r_path']
            p_path = job_df.loc[i, 'p_path']
        else:
            r_smiles = job_df.loc[i, 'product_smiles_am']
            p_smiles = job_df.loc[i, 'reactant_smiles_am']
            r_path = job_df.loc[i, 'p_path']
            p_path = job_df.loc[i, 'r_path']

        tarfile = job_df.loc[i, 'tarfile']

        qsub_name = qsub_prep_structure(i, reactant_idx, reaction_idx, letter,
                                        r_smiles, p_smiles, script_path, cpus,
                                        mem, tarfile, r_path, p_path)

        slurmid = os.popen("sbatch " + qsub_name).read()
        slurmid = int(slurmid.strip().split()[-1])

        submitted_jobs.add(slurmid)

        if len(submitted_jobs) >= max_queue:
            while True:
                job_info1 = os.popen("squeue -u mharris").readlines()[1:]
                current_jobs1 = {int(job.split()[0]) for job in job_info1}
                if len(current_jobs1) >= max_queue:
                    time.sleep(15)
                else:
                    finished_jobs = submitted_jobs - current_jobs1
                    print("finished jobs: ", finished_jobs)
                    # print("submitted jobs: ", submitted_jobs)
                    for job in finished_jobs:
                        submitted_jobs.remove(job)
                    # print("submitted jobs: ", submitted_jobs)
                    break
    while True:
        job_info = os.popen("squeue -u mharris").readlines()[1:]
        current_jobs = {int(job.split()[0]) for job in job_info}
        if len(current_jobs) > 0:
            time.sleep(15)
        else:
            break


if __name__ == "__main__":
    CSV_DF = sys.argv[1]
    DATAFRAME = pd.read_csv(CSV_DF)
    CPUS = 1
    MEM = "2GB"
    MAX_QUEUE = 300

    SCRIPT = 'rmsd_pp_barrier_estimate/rmsd_pp_no_slurm.py'
    run_calculations_structures(DATAFRAME, SCRIPT, CPUS, MEM,
                                MAX_QUEUE)
