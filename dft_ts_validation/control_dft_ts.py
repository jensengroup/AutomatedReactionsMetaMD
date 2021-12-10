#!/groups/kemi/mharris/.conda/envs/rdkit_2020_09/bin/python

import tarfile
import sys
import os
import textwrap
import time

import numpy as np
import pandas as pd


def qsub_prep(job_name, job_dir, script_path, interpolation_dir, rsmi, psmi,
              cpus, mem):
    """
    write qsub file for SLURM submission
    """
    pwd = os.getcwd()

    qsub_file = """\
    #!/bin/sh
    #SBATCH --job-name={0}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task={1}
    #SBATCH --mem={2}G
    #SBATCH --error={3}/{0}.stderr
    #SBATCH --output={3}/{0}.stdout
    #SBATCH --ntasks=1
    #SBATCH --time=100:00:00
    #SBATCH --partition=kemi1
    #SBATCH --no-requeue

    cp -r {8}/{0}/{5} /scratch/$SLURM_JOB_ID

    export GAUSS_SCRDIR=/scratch/$SLURM_JOB_ID

    cd /scratch/$SLURM_JOB_ID


    #run python code
    ({4} {5} '{6}' '{7}' {1} {2})

    cp -r ts_test_dft {3}/{8}/{0}/


    """.format(job_name, cpus, mem, pwd, script_path, interpolation_dir, rsmi,
               psmi, job_dir)

    with open(str(job_name) + "_qsub.tmp", "w") as qsub:
        qsub.write(textwrap.dedent(qsub_file))

    return str(job_name) + "_qsub.tmp"


def check_path_interpolation(directory):
    os.chdir(directory)

    files = [f for f in os.listdir(os.curdir) if
             f.endswith("xtbout")]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    max_energy = None
    for file_name in files:
        with open(file_name, 'r') as _file:
            line = _file.readline()
            while line:
                if 'TOTAL ENERGY' in line:
                    energy_au = np.float(line.split()[3])
                line = _file.readline()
        if not max_energy:
            max_energy = energy_au
            ts_guess = file_name[:-6]+'xyz'
        if energy_au > max_energy:
            max_energy = energy_au
            ts_guess = file_name[:-6]+'xyz'
    print(ts_guess, max_energy)

    os.chdir('../')
    return ts_guess, max_energy


def find_ts_guess(directory):
    """
    when sp calculations ar efinished: find the structure with maximum xtb
    energy
    """
    ts_guess_paths = []
    ts_guess_energies = []
    os.chdir(directory)
    high_temperature = False
    if os.path.exists('ht'):
        os.chdir('ht')
        high_temperature = True

    paths = [d for d in os.listdir(os.curdir) if d.startswith('path') and os.path.isdir(d)]
    print(paths)
    for path in paths:
        ts_guess, max_energy = check_path_interpolation(path)
        ts_guess_paths.append(path+'/'+ts_guess)
        ts_guess_energies.append(max_energy)

    ts_guess = ts_guess_paths[ts_guess_energies.index(max(ts_guess_energies))]
    if high_temperature:
        os.chdir('../')
        ts_guess = 'ht/'+ts_guess

    os.chdir('../../../')
    return ts_guess


def run_calculations(df, script_path, cpus, mem, max_queue):
    """
    For each reaction in the dataframe: do an xTB TS optimization with Gaussian
    optimizer. afterwards do IRC to check TS corresponds to intended reaction.
    Submits jobs when queue below max_queue
    """
    print(df)
    submitted_jobs = set()
    for job_name, reactant, r_idx, letter, path_idx in zip(df.index, df.reactant,
                                                           df.r_idx, df.letter,
                                                           df.path_idx):
        _dir = "{0}/{0}_{1}_{2}".format(reactant, r_idx, letter)
        os.mkdir(_dir+'_dft')
        tar = tarfile.open(_dir+'.tar.gz', 'r:gz')
        tar.extractall(path=_dir+'_dft')
        tar.close()

        if df.loc[job_name, 'pointAB']-df.loc[job_name, 'pointE'] > 3 and df.loc[job_name, 'pointE'] == 0:
            suffix = 'AB'
        elif df.loc[job_name, 'pointAB']-df.loc[job_name, 'pointE'] > 3 and df.loc[job_name, 'pointAB']/df.loc[job_name, 'pointE'] > 3:
            suffix = 'AB'
        else:
            suffix = 'E'
 
        if os.path.exists(_dir+'_dft/'+str(job_name)+'/ht'):
            interpolation_dir = 'ht/path'+suffix+str(path_idx)
        else:
            interpolation_dir = 'path'+suffix+str(path_idx)

        rsmi = df.loc[job_name, 'reactant_smiles_am']
        psmi = df.loc[job_name, 'product_smiles_am']
        qsub_name = qsub_prep(job_name, _dir+'_dft', script_path,
                              interpolation_dir, rsmi, psmi, cpus, mem)
        slurmid = os.popen("sbatch " + qsub_name).read()
        slurmid = int(slurmid.strip().split()[-1])

        submitted_jobs.add(slurmid)

        if len(submitted_jobs) >= max_queue:
            while True:
                job_info = os.popen("squeue -u mharris").readlines()[1:]
                current_jobs = {int(job.split()[0]) for job in job_info}
                if len(current_jobs) >= max_queue:
                    time.sleep(15)
                else:
                    finished_jobs = submitted_jobs - current_jobs
                    for job in finished_jobs:
                        submitted_jobs.remove(job)
                    break
    while True:
        job_info = os.popen("squeue -u mharris").readlines()[1:]
        current_jobs = {int(job.split()[0]) for job in job_info}
        if len(current_jobs) > 0:
            time.sleep(15)
        else:
            break


if __name__ == "__main__":

    df = pd.read_csv(sys.argv[1], index_col=0)
    #df_xtb_check = df[df.ts_barrier.notnull()]
    #df_xtb_check = df_xtb_check.reset_index()
    #df_xtb_check = pd.DataFrame(df_xtb_check.groupby(['reactant', 'r_idx']).first()).reset_index().set_index('index')
    print(df)
    script_path = '/groups/kemi/mharris/github/dft_ts_test/run_dft_ts.py'
    cpus = 4
    mem = 8
    max_queue = 200

    run_calculations(df, script_path, cpus, mem, max_queue)


