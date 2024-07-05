#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=phi3_mini
#SBATCH --mem=80G

module purge

module load Python/3.11.3-GCCcore-12.3.0

source /scratch/s4550684/venvs/phi3.3/bin/activate

input_directory="testset"
example_directory="fewshot"
output_directory="summout"

cp -r "$input_directory" $TMPDIR
cp -r "$example_directory" $TMPDIR
cp phi3_prompting.py $TMPDIR

cd $TMPDIR

mkdir "$output_directory"

python3 phi3_prompting.py "$example_directory" "$file_name"

cp -r "$output_directory" /scratch/s4550684/thesis
