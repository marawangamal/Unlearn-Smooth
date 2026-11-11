#!/bin/bash
#SBATCH --job-name=wmdp-npo-sam
#SBATCH --account=aip-grabus
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/tamia/logs/wmdp-unlearn-%j.out
set -euo pipefail

# Submit from examples/sam after preparing WMDP/files/data and WMDP/.cache.
# mkdir -p artifacts/tamia/logs
# SAM_ENV=/path/to/wmdp/environment sbatch WMDP/scripts/tamia/0_run_unlearning.sbatch
cd "$SLURM_SUBMIT_DIR"
source "${SAM_ENV:?Set SAM_ENV to the prepared WMDP environment}/bin/activate"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface-datasets-unlearn-smooth"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

artifacts=$(realpath artifacts)
run_name=${1:-wmdp-npo-sam-$SLURM_JOB_ID}
models="$artifacts/tamia/models/$run_name"
evals="$artifacts/tamia/evals/$run_name"
for method in npo npo-sam; do
    [[ ! -e "$models/$method" && ! -e "$evals/$method" ]]
    mkdir -p "$models/$method" "$evals/$method"
    ln -s "$models/$method" "$evals/$method/checkpoints"
done
cd WMDP

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO.json \
    --unlearn.lr=5e-06 \
    --unlearn.NPO+FT.beta=0.0225 \
    --unlearn.NPO+FT.gamma=1.0 \
    --overall.seed=1001 \
    --unlearn.max_steps=125 \
    --overall.cache_dir "$HF_HOME/hub" \
    --logger.name npo \
    --logger.json.root "$evals"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO+SAM.json \
    --unlearn.lr=5e-06 \
    --unlearn.NPO+FT+SAM.beta=0.015 \
    --unlearn.NPO+FT+SAM.gamma=2.25 \
    --unlearn.NPO+FT+SAM.sam_rho=0.01 \
    --overall.seed=1001 \
    --unlearn.max_steps=125 \
    --overall.cache_dir "$HF_HOME/hub" \
    --logger.name npo-sam \
    --logger.json.root "$evals"
