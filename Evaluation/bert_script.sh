#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=bertscore_eval
#SBATCH --mem=40G

module purge

module load Python/3.11.3-GCCcore-12.3.0

source /scratch/s4550684/venvs/phi3.3/bin/activate

candidates="summout"
references="closedllm"
output_dir="bertscore"

cp -r "$candidates" $TMPDIR
cp -r "$references" $TMPDIR
cp bertscore.py $TMPDIR

cd $TMPDIR

mkdir "$output_dir"

python3 bertscore.py

cp -r "$output_dir" /scratch/s4550684/thesis
