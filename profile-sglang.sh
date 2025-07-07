TP_SIZE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        [0-9]*)
            # 如果第一个参数是数字，直接作为 TP_SIZE
            if [ -z "$TP_SIZE" ]; then
                TP_SIZE="$1"
                shift
            else
                echo "unknown parameter: $1"
                exit 1
            fi
            ;;
        *)
            echo "unknown parameter: $1"
            exit 1
            ;;
    esac
done

# check if TP_SIZE is provided
if [ -z "$TP_SIZE" ]; then
    echo "error: please provide tensor_model_parallel_size value"
    echo "usage: bash profile-sglang.sh [--tp] <size>"
    echo "example: bash profile-sglang.sh 2"
    echo "example: bash profile-sglang.sh --tp 2"
    exit 1
fi

mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# truncate now's return value, only keep date, hour and minute (format: DD-HH-MM)
TIMESTAMP=$(date +"%d-%H-%M")

echo "TP_SIZE: ${TP_SIZE}"

echo "TIMESTAMP: ${TIMESTAMP}"

source ~/.python/veRL-multiturn-rollout/bin/activate

COMMAND="nohup bash examples/grpo_trainer/run_qwen2-7b_seq_balance.sh trainer.experiment_name=qwen2-7b_rm-gsm8k-grpo-seq-balance-tp-${TP_SIZE}-${TIMESTAMP} --tp ${TP_SIZE} >> logs/gsm8k-tp-${TP_SIZE}-${TIMESTAMP}.log 2>&1 &"

run_command() {
    echo "Running command: $1"
    eval "$1"
}

run_command "${COMMAND}"

alias kill="pkill -f sglang"

alias tf="tail -f"