#!/bin/bash

# Specify the path to the config file
config=./results/frame_tie/$1_overlap_frame_tie.txt
PSR_name="$1"

# Check if the file exists
if [[ ! -f "$config" ]]; then
    echo "Error: Config file $config does not exist."
    exit 1
fi

# Get number of lines, ensuring a valid job array
n_lines=$(wc -l < "$config")
n_lines=$((n_lines - 2))

if [[ $n_lines -lt 0 ]]; then
    echo "Error: Invalid job array size. The file $config may have insufficient lines."
    exit 1
fi

# Generate the correct Slurm array specification
if [[ $n_lines -gt 0 ]]; then
    array_spec="--array=0-${n_lines}"
else
    echo "Error: No valid jobs to submit. Check your input file."
    exit 1
fi

# Generate the Slurm script with the correct array size
cat <<EOF > job_script.sh
#!/bin/bash -l

#SBATCH --job-name=VLBI
#SBATCH --account=vlbi
#SBATCH --partition=tier3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=0-20:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#SBATCH ${array_spec}

conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate VLBI

# Extract parameters for the current SLURM_ARRAY_TASK_ID
RAJ=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$2}' $config)
DECJ=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$3}' $config)
PMRA=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$4}' $config)
PMDEC=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$5}' $config)
PX=\$(awk -v ArrayTaskID=\${SLURM_ARRAY_TASK_ID} '\$1==ArrayTaskID {print \$6}' $config)

PSR_name="${PSR_name}"  # Ensure variable is correctly passed

echo "\${PSR_name}, \${SLURM_ARRAY_TASK_ID}, RAJ = \${RAJ}, DECJ = \${DECJ}, PMRA = \${PMRA}, PMDEC = \${PMDEC}, PX \${PX}." >> output.txt

srun --mem-per-cpu=10g python3 -u calculate_posterior.py \${PSR_name} \${SLURM_ARRAY_TASK_ID} \${RAJ} \${DECJ} \${PMRA} \${PMDEC} \${PX}

EOF

# Submit the job
sbatch job_script.sh
