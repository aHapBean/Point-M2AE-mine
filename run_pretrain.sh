export MASTER_ADDR="localhost"  # CropMAE needs this
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
JOB_NAME='test'
GPUS=${GPUS:-1}        # 如果没设置GPS则默认64 export GPUS=128
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}
srun --partition=MoE \
        --mpi=pmi2 \
        --gres=gpu:${GPUS_PER_NODE} \
        -n1 --ntasks-per-node=${GPUS_PER_NODE} \
        --job-name=${JOB_NAME} \
        --kill-on-bad-exit=1 \
        --nodes=1 \
        --ntasks=${GPUS} \
        --quotatype=reserved \
        python main.py --config cfgs/pre-training/point-m2ae.yaml --exp_name test