source openr1/bin/activate
cd /workspace/rlpvr-open-r1/

# export MODEL_PATH=/dev/shm/checkpoint-1641
# export SAVE_NAME=qwen-2.5-math-7b
MODEL_PATH=$1
SAVE_NAME=$2

export C3S_KEYTAB=/workspace/rlpvr-open-r1/c3s.search-gpt.keytab 
export C3S_ACCOUNT=search-gpt 
kinit -kt ${C3S_KEYTAB} ${C3S_ACCOUNT}@C3.NAVER.COM 
klist -kte ${C3S_KEYTAB} 
export HDFS_CONNECTOR_PATH=/root/c3s-hdfs-connector-0.7/bin/hdfs-connector


$HDFS_CONNECTOR_PATH -ls hdfs://jmt/user/search-gpt/models/multimodal/datasets
# download 
mkdir datasets
$HDFS_CONNECTOR_PATH -get -f hdfs://jmt/user/search-gpt/models/multimodal/datasets/ ./

# Eval
mkdir -p /workspace/rlpvr-open-r1/evaluation/outputs/math_data/aime_2024/
mkdir -p /workspace/rlpvr-open-r1/evaluation/outputs/math_data/gsm8k/
mkdir -p /workspace/rlpvr-open-r1/evaluation/outputs/math_data/math_500/
mkdir -p /workspace/rlpvr-open-r1/evaluation/outputs/knowledge_data/gpqa_questions/
mkdir -p /workspace/rlpvr-open-r1/evaluation/outputs/knowledge_data/mmlu_questions/
mkdir -p /workspace/rlpvr-open-r1/evaluation/outputs/if_data/question_arena_hard/
mkdir -p /workspace/rlpvr-open-r1/evaluation/outputs/if_data/questions_alpaca_eval/
mkdir -p /workspace/rlpvr-open-r1/evaluation/outputs/if_data/question_mt_bench/

# math

python /workspace/rlpvr-open-r1/evaluation/predict_vllm_math.py \
--model_path $MODEL_PATH \
--data_path /workspace/rlpvr-open-r1/datasets/eval/math_data/gsm8k_4x.json \
--output_path /workspace/rlpvr-open-r1/evaluation/outputs/math_data/gsm8k/${SAVE_NAME}.csv

python /workspace/rlpvr-open-r1/evaluation/predict_vllm_math.py \
--model_path $MODEL_PATH \
--data_path /workspace/rlpvr-open-r1/datasets/eval/math_data/aime_2024_64x.json \
--output_path /workspace/rlpvr-open-r1/evaluation/outputs/math_data/aime_2024/${SAVE_NAME}.csv

python /workspace/rlpvr-open-r1/evaluation/predict_vllm_math.py \
--model_path $MODEL_PATH \
--data_path /workspace/rlpvr-open-r1/datasets/eval/math_data/math_500_8x.json \
--output_path /workspace/rlpvr-open-r1/evaluation/outputs/math_data/math_500/${SAVE_NAME}.csv


# knowledge
python /workspace/rlpvr-open-r1/evaluation/predict_vllm_knowledge.py \
--model_path $MODEL_PATH \
--data_path /workspace/rlpvr-open-r1/datasets/eval/knowledge_data/gpqa_questions_16x.jsonl \
--output_path /workspace/rlpvr-open-r1/evaluation/outputs/knowledge_data/gpqa_questions/${SAVE_NAME}.jsonl

python /workspace/rlpvr-open-r1/evaluation/predict_vllm_knowledge.py \
--model_path $MODEL_PATH \
--data_path /workspace/rlpvr-open-r1/datasets/eval/knowledge_data/mmlu_questions.jsonl \
--output_path /workspace/rlpvr-open-r1/evaluation/outputs/knowledge_data/gpqa_questions/${SAVE_NAME}.jsonl

# alpaca eval, arena hard, mt bench
python /workspace/rlpvr-open-r1/evaluation/predict_vllm_arena_hard.py \
--model_path $MODEL_PATH \
--data_path /workspace/rlpvr-open-r1/datasets/eval/if_data/question_arena_hard.jsonl \
--output_path /workspace/rlpvr-open-r1/evaluation/outputs/question_arena_hard/${SAVE_NAME}.jsonl \
--model_name ${SAVE_NAME}

python /workspace/rlpvr-open-r1/evaluation/predict_vllm_alpaca_eval.py \
--model_path $MODEL_PATH \
--data_path /workspace/rlpvr-open-r1/datasets/eval/if_data/questions_alpaca_eval.jsonl \
--output_path /workspace/rlpvr-open-r1/evaluation/outputs/questions_alpaca_eval/${SAVE_NAME}.jsonl \
--model_name ${SAVE_NAME}

python /workspace/rlpvr-open-r1/evaluation/predict_vllm_mt_bench.py \
--model_path $MODEL_PATH \
--data_path /workspace/rlpvr-open-r1/datasets/eval/if_data/question_mt_bench.jsonl \
--output_path /workspace/rlpvr-open-r1/evaluation/outputs/question_mt_bench/${SAVE_NAME}.jsonl \
--model_name ${SAVE_NAME}