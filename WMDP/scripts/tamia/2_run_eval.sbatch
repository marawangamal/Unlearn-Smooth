#!/bin/bash
#SBATCH --job-name=wmdp-eval
#SBATCH --account=aip-grabus
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/tamia/logs/wmdp-eval-%j.out
set -euo pipefail

# Submit from examples/sam after unlearning and relearning complete.
# SAM_ENV=/path/to/wmdp/environment sbatch WMDP/scripts/tamia/2_run_eval.sbatch RUN_NAME
cd "$SLURM_SUBMIT_DIR"
source "${SAM_ENV:?Set SAM_ENV to the prepared WMDP environment}/bin/activate"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface-datasets-unlearn-smooth"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

artifacts=$(realpath artifacts)
run_name=${1:?Pass the matching unlearning run name}
models="$artifacts/tamia/models/$run_name"
evals="$artifacts/tamia/evals/$run_name"
methods=(npo npo-sam)
for method in npo npo-sam; do
    for samples in 20 40 60; do
        methods+=("$method-n$samples-m1-seed1")
    done
done
for method in "${methods[@]}"; do
    [[ -f "$models/$method/config.json" ]]
done
cd WMDP

for method in "${methods[@]}"; do
    accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
        --mixed_precision no -m lm_eval --model hf \
        --model_args "pretrained=$models/$method,cache_dir=./.cache" \
        --tasks mmlu,wmdp_bio \
        --batch_size auto \
        --output_path "$evals/$method/evaluation-$SLURM_JOB_ID"
done
