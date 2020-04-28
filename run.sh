#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1
#SBATCH -o ./$1.$2-%A.out
#SBATCH -e ./$1.$2-%A.err
#SBATCH --nodelist=$3
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -c8
#SBATCH --mem=180G
#SBATCH --time=4-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate python3.7

set PYTHONPATH=./

# Start the experiment.
python -m torch.distributed.launch --nproc_per_node=4 ./$1/Run.py --mode='$2' --data_path='$4' --dataset='$5'
EOT
