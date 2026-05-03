#!/usr/bin/env bash
set -euo pipefail

time=$(date +%F-%H:%M)
start_time="${time}"
echo "All experiments started at ${start_time}."
# 8 张 GPU
GPUS=(0 1 2 3 4 5 6 7)

# 数据集与配置目录
CONFIG_ROOT="config/classic_cv"
DATASET="utkface"

# 标签数量
LABELS=(10 50 250 2000)

# seed 设置
SEEDS=(0)
# 如果要跑 3 个 seed，改成：
# SEEDS=(0 1 2)

# fullysupervised 单独处理，因为它不带 lb
FULLY_ALG="fullysupervised"

# 需要带 label 的算法
ALGS=(
  "supervised"
  "pimodel"
  "meanteacher"
  "ucvme"
  "clss"
  "mixmatch"
  "rankup"
  "rankuprda"
)

# 构造任务列表
JOBS=()

for seed in "${SEEDS[@]}"; do
  # fullysupervised: 不带 label
  cfg="${CONFIG_ROOT}/${FULLY_ALG}/${FULLY_ALG}_${DATASET}_s${seed}.yaml"
  log_dir="./outputs_log/${DATASET}/${FULLY_ALG}"
  log_file="${time}_s${seed}.log"
  JOBS+=("${cfg}|${log_dir}|${log_file}")

  # 其他算法: 带 label
  for alg in "${ALGS[@]}"; do
    for lb in "${LABELS[@]}"; do
      cfg="${CONFIG_ROOT}/${alg}/${alg}_${DATASET}_lb${lb}_s${seed}.yaml"
      log_dir="./outputs_log/${DATASET}/${alg}"
      log_file="${time}_lb${lb}_s${seed}.log"
      JOBS+=("${cfg}|${log_dir}|${log_file}")
    done
  done
done

echo "Total jobs: ${#JOBS[@]}"

run_worker() {
  local gpu="$1"
  local worker_id="$2"
  local num_workers="$3"

  for ((i=worker_id; i<${#JOBS[@]}; i+=num_workers)); do
    IFS="|" read -r cfg log_dir log_file <<< "${JOBS[$i]}"

    mkdir -p "${log_dir}"

    if [[ ! -f "${cfg}" ]]; then
      echo "[WARN] Config not found, skip: ${cfg}" | tee -a "./outputs_log/${DATASET}/missing_config.log"
      continue
    fi

    echo "[GPU ${gpu}] Start: ${cfg}"
    echo "[GPU ${gpu}] Log: ${log_dir}/${log_file}"

    python train.py --gpu "${gpu}" --c "${cfg}" > "${log_dir}/${log_file}" 2>&1

    echo "[GPU ${gpu}] Done: ${cfg}"
  done
}

# 启动 8 个 worker，每张 GPU 一个 worker
for idx in "${!GPUS[@]}"; do
  run_worker "${GPUS[$idx]}" "${idx}" "${#GPUS[@]}" &
done

wait

end_time=$(date +%F-%H:%M)
echo "All experiments finished at ${end_time}."