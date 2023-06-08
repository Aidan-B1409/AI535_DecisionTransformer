#!/bin/bash
#SBATCH -w dgx2-3
#SBATCH -p dgx
#SBATCH -A eecs
#SBATCH --job-name=decision_transformer
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:6
#SBATCH --mem=20G
#SBATCH --cpus-per-task 2
#SBATCH --export=ALL

source /nfs/guille/eecs_research/soundbendor/beerya/miniconda3/bin/activate
source activate decisiontransformer
python3 src/app.py 0.9 -e FetchPick -t True
