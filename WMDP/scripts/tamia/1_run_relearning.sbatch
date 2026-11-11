#!/bin/bash
#SBATCH --job-name=wmdp-relearn
#SBATCH --account=aip-grabus
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/tamia/logs/wmdp-relearn-%j.out
set -euo pipefail

# Submit from examples/sam after the matching unlearning job completes.
# SAM_ENV=/path/to/wmdp/environment sbatch WMDP/scripts/tamia/1_run_relearning.sbatch RUN_NAME
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
[[ -f "$models/npo/config.json" && -f "$models/npo-sam/config.json" ]]
for method in npo-n60-m1-seed1 npo-sam-n60-m1-seed1; do
    [[ ! -e "$models/$method" && ! -e "$evals/$method" ]]
    mkdir -p "$models/$method" "$evals/$method"
    ln -s "$models/$method" "$evals/$method/checkpoints"
done
cd WMDP

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/relearn_model.py \
    --config-file configs/unlearn/wmdp/Relearn+Forget.json \
    --overall.model_name "$models/npo" \
    --unlearn.max_steps=15 \
    --overall.cache_dir "$HF_HOME/hub" \
    --logger.name npo-n60-m1-seed1 \
    --logger.json.root "$evals"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/relearn_model.py \
    --config-file configs/unlearn/wmdp/Relearn+Forget.json \
    --overall.model_name "$models/npo-sam" \
    --unlearn.max_steps=15 \
    --overall.cache_dir "$HF_HOME/hub" \
    --logger.name npo-sam-n60-m1-seed1 \
    --logger.json.root "$evals"
