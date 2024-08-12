#!/bin/bash

# Specify the path to the config file
config=./results/$1_overlap.txt
n_lines=$(wc -l < $config)
n_lines=$((n_lines - 2))

# Generate the Slurm script with the correct array size
cat <<EOF > job_script.sh
#!/bin/bash -l

#SBATCH --job-name=VLBI         # Name of your job
#SBATCH --account=vlbi          # Your Slurm account
#SBATCH --partition=debug       # Run on tier3
#SBATCH --output=%x_%j.out      # Output file
#SBATCH --error=%x_%j.err       # Error file
#SBATCH --time=1-12:10:00       # 10 minute time limit
#SBATCH --ntasks=1              # 1 tasks (i.e. processes)
#SBATCH --mem-per-cpu=1g        # 1GB RAM per CPU
#SBATCH --array=0-${n_lines}    # Array size

echo "SLURM_ARRAY_TASK_ID = \${SLURM_ARRAY_TASK_ID}"

# Extract the PMRA for the current SLURM_ARRAY_TASK_ID
PMRA=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$2}' $config)

# Extract the PMDEC for the current SLURM_ARRAY_TASK_ID
PMDEC=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$3}' $config)

# Extract the PX for the current SLURM_ARRAY_TASK_ID
PX=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$4}' $config)

echo "\${SLURM_ARRAY_TASK_ID}, PMRA = \${PMRA}, PMDEC = \${PMDEC}, PX \${PX}." >> output.txt

srun python3 -u calculate posterior.py \$1 \${SLURM_ARRAY_TASK_ID} \${PMRA} \${PMDEC} \${PX}

EOF

# Submit the job
sbatch job_script.sh
