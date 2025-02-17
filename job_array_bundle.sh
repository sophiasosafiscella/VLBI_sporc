# Specify the path to the config file
config=./results/frame_tie/$1_overlap_frame_tie.txt
PSR_name="$1"
n_lines=$(wc -l < "$config")
n_lines=$((n_lines - 1))  # Adjust for the header (subtract 1 instead of 2)

# Get the maximum allowed job array size
MaxArraySize=10001  # Replace this with `scontrol show config | grep MaxArraySize | awk '{print $NF}'` if needed

if (( n_lines > MaxArraySize )); then
    bundle_mode=true
    lines_per_job=$(( (n_lines + MaxArraySize - 1) / MaxArraySize ))  # Ensures full coverage
    array_size=$(( (n_lines + lines_per_job - 1) / lines_per_job ))
else
    bundle_mode=false
    array_size=$n_lines
fi

# Generate the Slurm script dynamically
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
#SBATCH --array=0-${array_size}

conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate VLBI

config="${config}"
PSR_name="${PSR_name}"

if [[ "$bundle_mode" == "true" ]]; then
    start=\$(( SLURM_ARRAY_TASK_ID * $lines_per_job + 2 ))  # Start at line 2
    end=\$(( start + $lines_per_job - 1 ))
    [[ \$end -gt $((n_lines + 1)) ]] && end=$((n_lines + 1))  # Account for header
else
    start=\$(( SLURM_ARRAY_TASK_ID + 2 ))  # Adjust for zero-based index and header
    end=\$start
fi

# Read only from line 2 onwards
awk "NR>=\$start && NR<=\$end" "\$config" | while read -r ArrayTaskID RAJ DECJ PX PMRA PMDEC POSEPOCH; do
    echo "\${PSR_name}, \${ArrayTaskID}, RAJ = \${RAJ}, DECJ = \${DECJ}, PX = \${PX}, PMRA = \${PMRA}, PMDEC = \${PMDEC}, POSEPOCH = \${POSEPOCH}." >> output.txt

    srun --mem-per-cpu=10g python3 -u calculate_posterior.py "\${PSR_name}" "\${ArrayTaskID}" "\${RAJ}" "\${DECJ}" "\${PX}" "\${PMRA}" "\${PMDEC}" "\${POSEPOCH}"
done

EOF

# Submit the job!
sbatch job_script.sh
