#!/bin/bash
set -euo pipefail

# 运行所有网络/细胞类型/TF配置的数据集，支持多卡并发（每卡一次只跑一个数据集）。
# 用法示例：
#   bash run.sh                                        # 默认四卡、所有网络、TFs_500、默认种子
#   bash run.sh --gpus "0,1"                          # 指定卡运行
#   bash run.sh --network_type Specific               # 只运行Specific网络
#   bash run.sh --network_type STRING                 # 只运行STRING网络
#   bash run.sh --tf_config TFs_1000                  # 指定 TF 过滤
#   bash run.sh --seeds "42,66,80"                    # 指定种子运行多轮实验
#   bash run.sh --gpus "0,2" --batch_size 128          # 透传 demo 基础超参
# GPU 接口：--gpus "0,1,2"（默认 "0,1,2,3"）
# 网络类型接口：--network_type Specific（默认运行所有网络）
# TF 配置接口：--tf_config TFs_500（默认）
# 种子接口：--seeds "42,66,80,12,100"（默认 "42,66,80,12,100,10,20,30,40,50"）
# 训练参数接口（对应 scripts/demo.py 基础参数）：--batch_size --embed_size --num_layers --num_head --lr --epochs --step_size --gamma --scheduler_flag --patience --loss_type

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT_DIR}/data/Dataspilt"
DEMO="${ROOT_DIR}/scripts/demo.py"

# 默认值
GPU_IDS="0,1,2,3"
NETWORK_FILTER=""  # 空字符串表示运行所有网络
TF_FILTER="TFs_1000"
# SEEDS="42,66,80,12,100,10,20,30,40,50"
SEEDS="10,20,30,40,50"
RESULTS_PATH="${RESULTS_PATH:-yourpath/result/summary.csv}"
DEMO_ARGS=()
MAX_JOBS_PER_GPU=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus|--num_gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --tf_config)
      TF_FILTER="$2"
      shift 2
      ;;
    --network_type)
      NETWORK_FILTER="$2"
      shift 2
      ;;
    --max_jobs_per_gpu)
      MAX_JOBS_PER_GPU="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --batch_size|--embed_size|--num_layers|--num_head|--lr|--epochs|--step_size|--gamma|--scheduler_flag|--patience|--loss_type|--subgraph_strategy|--use_globalgraph|--globalgraph_gnn_layers|--subgraph_gnn_layers|--gnn_type|--gnn_hidden|--gnn_heads|--gnn_dropout|--subgraph_hops|--max_subgraph_size|--use_subgraph)
      DEMO_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "可用参数: --gpus \"0,1,2\" --network_type Specific --tf_config TFs_500 --seeds \"42,66,80\" 以及 demo 的基础训练参数"
      exit 1
      ;;
  esac
done

# 解析种子列表
IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS"
NUM_SEEDS=${#SEEDS_ARR[@]}
if [[ $NUM_SEEDS -eq 0 ]]; then
  echo "种子列表为空，请用 --seeds 指定，例如 --seeds \"42,66,80\""
  exit 1
fi

# 解析GPU列表
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARR[@]}
if [[ $NUM_GPUS -eq 0 ]]; then
  echo "GPU 列表为空，请用 --gpus 指定，例如 --gpus \"0,1,2\""
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "数据根目录不存在: $DATA_ROOT"
  exit 1
fi

# 查找所有数据集
mapfile -t DATASETS < <(find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -type d -name "TFs_*" | sort)

# 按网络类型过滤
if [[ -n "$NETWORK_FILTER" ]]; then
  mapfile -t DATASETS < <(printf "%s\n" "${DATASETS[@]}" | awk -v net="$NETWORK_FILTER" 'BEGIN{FS="/"} {if($(NF-2)==net) print $0}')
fi

# 按 TF 配置过滤
if [[ -n "$TF_FILTER" ]]; then
  mapfile -t DATASETS < <(printf "%s\n" "${DATASETS[@]}" | awk -v tf="$TF_FILTER" 'END{if(NR==0) exit} {n=split($0,a,"/"); if(a[n]==tf) print $0}')
fi
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "未在 $DATA_ROOT 下找到任何 TFs_* 数据集目录"
    exit 1
fi

echo "Starting scGPL batch runs over datasets in $DATA_ROOT"
echo "GPU list: ${GPU_IDS}"
echo "Max concurrency: ${MAX_JOBS_PER_GPU} per GPU"
echo "Network type filter: ${NETWORK_FILTER:-all networks}"
echo "TF config filter: ${TF_FILTER}"
echo "Seeds to run: ${SEEDS} (${NUM_SEEDS} seeds)"
echo "Training parameters: ${DEMO_ARGS[*]:-default}"
echo "Number of datasets: ${#DATASETS[@]}"
printf "First 3 datasets:\n%s\n" "$(printf '%s\n' "${DATASETS[@]:0:3}")"
echo "Total tasks to run: $((${#DATASETS[@]} * ${NUM_SEEDS}))"

# 生成所有任务列表：所有(seed, dataset)组合
ALL_TASKS=()
for seed in "${SEEDS_ARR[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    ALL_TASKS+=("${seed}:${dataset}")
  done
done

echo "=========================================="
echo "Starting unified scheduling across all seeds and datasets"
echo "=========================================="

# 全局调度参数
declare -A GPU_JOB_COUNT
for gpu in "${GPU_ARR[@]}"; do
  GPU_JOB_COUNT[$gpu]=0
done
RUNNING_JOBS=() # 元素: pid:gpu:seed:dataset

start_task() {
  local task="$1"
  local gpu="$2"
  IFS=':' read -r seed dataset <<< "$task"
  local dataset_subpath="${dataset#$DATA_ROOT/}"
  echo "[START] GPU ${gpu} -> ${dataset_subpath} (seed: ${seed})"

  # 设置环境变量传递种子
  export SEED="$seed"

  CUDA_VISIBLE_DEVICES="${gpu}" python "${DEMO}" --data_dir "${dataset}" --seed "${seed}" "${DEMO_ARGS[@]}" &
  local pid=$!
  RUNNING_JOBS+=("${pid}:${gpu}:${seed}:${dataset_subpath}")
  GPU_JOB_COUNT[$gpu]=$(( GPU_JOB_COUNT[$gpu] + 1 ))
}

check_running_jobs() {
  local new_running=()
  for job in "${RUNNING_JOBS[@]}"; do
    IFS=':' read -r pid gpu seed ds <<< "$job"
    if kill -0 "$pid" 2>/dev/null; then
      new_running+=("$job")
    else
      GPU_JOB_COUNT[$gpu]=$(( GPU_JOB_COUNT[$gpu] - 1 ))
      echo "[DONE ] GPU ${gpu} completed: ${ds} (seed: ${seed})"
    fi
  done
  RUNNING_JOBS=("${new_running[@]}")
}

  get_available_gpu() {
    for gpu in "${GPU_ARR[@]}"; do
      if [[ ${GPU_JOB_COUNT[$gpu]} -lt $MAX_JOBS_PER_GPU ]]; then
        echo "$gpu"
        return
      fi
    done
    echo "none"
  }

  print_status() {
    status=()
    for g in "${GPU_ARR[@]}"; do
      status+=("GPU${g}:${GPU_JOB_COUNT[$g]}/${MAX_JOBS_PER_GPU}")
    done
    echo "[STAT] $(IFS=' | '; echo "${status[*]}") | Running: ${#RUNNING_JOBS[@]}"
  }

TASK_INDEX=0
TOTAL_TASKS=${#ALL_TASKS[@]}
echo "[INFO] Starting unified scheduling, total tasks: ${TOTAL_TASKS}"

while [[ $TASK_INDEX -lt $TOTAL_TASKS || ${#RUNNING_JOBS[@]} -gt 0 ]]; do
  # 回收已完成任务
  check_running_jobs

  # 尝试启动新任务
  while [[ $TASK_INDEX -lt $TOTAL_TASKS ]]; do
    avail_gpu=$(get_available_gpu)
    if [[ "$avail_gpu" == "none" ]]; then
      break
    fi
    task="${ALL_TASKS[$TASK_INDEX]}"
    start_task "$task" "$avail_gpu"
    TASK_INDEX=$(( TASK_INDEX + 1 ))
    print_status
    sleep 1
  done

  # 没有可启动的新任务时等待一会儿
  if [[ ${#RUNNING_JOBS[@]} -gt 0 ]]; then
    sleep 5
  fi
done

echo "[ALL] Completed all tasks across all seeds and datasets"

echo "All scGPL experiments completed. Results appended to $RESULTS_PATH"
echo "Total tasks completed: $((${#DATASETS[@]} * ${NUM_SEEDS}))"
