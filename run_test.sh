source openr1/bin/activate

LOCALRANK=$1 # 0~7
N_MACHINES=$2 # not rank, total number of machines
N_GPUS=$3 # 8
MASTER_ADDR=$4-worker-0 # master, worker 앞 prefix
MASTER_PORT=29500

DISTRIBUTED_ARGS="
--machine_rank $LOCALRANK
--num_processes $N_GPUS
--num_machines $N_MACHINES
--main_process_ip $MASTER_ADDR
--main_process_port $MASTER_PORT
"

ACCELERATE_LOG_LEVEL=info accelerate launch $DISTRIBUTED_ARGS --config_file recipes/accelerate_configs/zero2_multi.yaml \
  src/open_r1/grpo.py \
  --config recipes/exp_configs/qwen-2.5-math-7b.yaml