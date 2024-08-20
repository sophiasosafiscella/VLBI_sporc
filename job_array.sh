#!/bin/bash -l

# Specify the path to the config file
config=./results/$1_overlap.txt
PSR_name="$1"
n_lines=$(wc -l < $config)
n_lines=$((n_lines - 2))
n_lines=1


# Generate the Slurm script with the correct array size
cat <<EOF > job_script.sh
#!/bin/bash -l

#SBATCH --job-name=VLBI         # Name of your job
#SBATCH --account=vlbi          # Your Slurm account
#SBATCH --partition=tier3       # Run on tier3
#SBATCH --output=%x_%A_%a.out      # Output file
#SBATCH --error=%x_%A_%a.err       # Error file
#SBATCH --time=0-01:00:00       # 10 minute time limit
#SBATCH --ntasks=1              # 1 tasks (i.e. processes)
#SBATCH --mem-per-cpu=4g        # 1GB RAM per CPU
#SBATCH --array=0-${n_lines}    # Array size

conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate VLBI

# Extract the PMRA for the current SLURM_ARRAY_TASK_ID
PMRA=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$2}' $config)

# Extract the PMDEC for the current SLURM_ARRAY_TASK_ID
PMDEC=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$3}' $config)

# Extract the PX for the current SLURM_ARRAY_TASK_ID
PX=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$4}' $config)

PSR_name="${PSR_name}"  # Correctly pass the variable into the script

echo "\${PSR_name}, \${SLURM_ARRAY_TASK_ID}, PMRA = \${PMRA}, PMDEC = \${PMDEC}, PX \${PX}." >> output.txt

srun --mem-per-cpu=4g python3 -u calculate_posterior_minimal.py \${PSR_name} \${SLURM_ARRAY_TASK_ID} \${PMRA} \${PMDEC} \${PX}

EOF

# Submit the job
sbatch job_script.sh
