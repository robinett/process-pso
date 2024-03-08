#!/bin/bash
#SBATCH --job-name=step_3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=/shared/pso/step_3_process_gleam/logs/sbatch_output_%j.log
#SBATCH --partition=catch-m6a4xl-spot

echo "running!"
# unload numpy module because it conflicts with poetry
module purge
# load the module
echo "modules loading!"
source /shared/poetry_pkgs/step-1-wYhrH9Mr-py3.8/bin/activate
# run the python code
python3 -u main.py
echo "done!"
