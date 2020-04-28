#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1
#SBATCH -o ./$1.eval-%A.out
#SBATCH -e ./$1.eval-%A.err
#SBATCH --nodelist=$2
#SBATCH -p cpu
#SBATCH -c8
#SBATCH --mem=50G
#SBATCH --time=4-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate python3.7

set PYTHONPATH=./

# Start the experiment.
python Run_Evaluation.py --model=$1 --data_path='$2'
EOT
