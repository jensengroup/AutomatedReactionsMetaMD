# Fast and automated identification of reactions with low barriers using meta-MD simulations

## Meta-MD product search procedure

### Usage 
Procedure to run meta-dynamics search for potential one-step reaction products:
* create .csv file containing the list of mapped reactant SMILES to be investigated
* change parameters CPUS, MEM, MAX_QUEUE, N_RUNS, S_FACTOR, TIME_PS, K_PUSH, ALP, INITIALIZE_PRODUCTS and SCRIPT in ```control_metadyn_runs.py```
  * CPUS: number of cpus each meta-dynamics job can use
  * MEM: amount of memory each meta-dynamics job can use
  * MAX_QUEUE: maximally allowed number of jobs to be submitted to ```Slurm``` at the same time
  * N_RUNS: number of meta-dynamics runs to be done for each reactant SMILES
  * S_FACTOR: the scaling parameter for the wall potential
  * TIME_PS: How often to check for change in adjacency matrix in ps
  * K_PUSH: the k_push meta-dynamics parameter
  * ALP: parameter controlling the width of the Gaussian meta-dynamics potential
  * INITILIZE_PRODUCTS: Should the biasing potential be initialized with products from previous run? 
  * SCRIPT: the path to ```reaction_box_no_slurm.py```
* change the path to xTB in ```reaction_box_no_slurm.py```


### External Programs:
The procedure relies on ```xTB``` for the quantum chemical calculations and ```Slurm``` for job submission
Also, a local version of ```xyz2mol``` (```xyz2mol_local.py```) is used to analyze the structures on the trajectory.
```
https://github.com/jensengroup/xyz2mol
```


Finally, the search is executed by:
```
./control_metadyn_runs.py reactant_smiles.csv
```

A folder named ```$S_FACTOR_$K_PUSH_$ALP``` is created, where the results are saved
Results are saved in  as dataframes in .pkl files: 1 for each run

The results for all runs can be combined using ```combine_runs.py```, which collects all recorded reactions and extracts all 1-step reactions, 
both are saved as .csv files.


## RMSD-PP procedure

If products are from the meta-dynamics product search, reactant and product structures can be extracted from the meta-dynamics
search. This option is set by calling --use-structures when creating the submission file in ```rmsd_pp_no_slurm.py```.
Otherwise structures are embedded using ```RDKit```. 

### Usage

A .csv file containing the reactions to get barrier estimates mus be created: these reactions can stem from either 
meta-dynamics search, systematic search or something else entirely.
The .csv file must contain one row for each reaction with at least two columns: 
* ```reactant_smiles_am``` containing the mapped SMILES of the reactant
* ```product_smiles_am``` containing the mapped SMILES of the product
Then change parameters CPUS, MEM, MAX_QUEUE and SCRIPT in ```control_rmsd_paths.py``` 
* CPUS: number of cpus each meta-dynamics job can use
* MEM: amount of memory each meta-dynamics job can use
* MAX_QUEUE: maximally allowed number of jobs to be submitted to ```Slurm``` at the same time
* SCRIPT: the path to ```rmsd_pp_no_slurm.py```

#### for meta-dyanmics products
```create_job_names.py``` can be used to make the reaction.csv file after meta-dynamics runs. 
By default 5 entries per reaction is created.
RXN_INDEX_LIST must be changed to the list of reactant indexes used in the meta-dynamics search


The RMSD-PP paths are then computed by running
```
./control_rmsd_paths.py reactions.csv
```

### External Programs:
The procedure relies on ```xTB``` for the quantum chemical calculations and ```Slurm``` for job submission
Also, a local version of ```xyz2mol``` (```xyz2mol_local.py```) is used to analyze the structures on the trajectory.
```
https://github.com/jensengroup/xyz2mol
```

Results for each run is saved in a .pkl file named according to the job name.

The results for all runs can be collected by 
```
./collect_dataframes.py reactions.csv
```
which returns an updated .csv file

Files for the RMSD-PP barrier estmation procedure is found in ```rmsd_pp_barrier_estimate``` 

## TS validation procedure
This procedures optimizes a TS based on the specified DFT functional and basis set and guess structures from the RMSD-PP procedure.

### Usage
As input is needed the results from the RMSD-PP procedure: both the paths containing the TS guesses, and the updated .csv file containing barrier estimates for the reactions. 

Change the parameters cpus, mem, max_queue and script in ```control_dft_ts.py``` to fit your requirements
* cpus: number of cpus each job should use
* mem: amount of memory each job should use
* max_queue: maximally allowed number of submitted jobs at a time
* script_path: path to ```run_dft_ts.py```

remember to change the paths to xTB and Gaussian.


### External Programs:
The procedure relies on ```Gaussian``` for the quantum chemical calculations and ```Slurm``` for job submission
Also, a local version of ```xyz2mol``` (```xyz2mol_local.py```) is used to analyze the structures on the trajectory.
```
https://github.com/jensengroup/xyz2mol
```

The calculations are done by calling
```
./control_dft_ts.py rmsd_output.csv
```
