# sagittal_classification.sh
#!/bin/bash
set -euo pipefail

# ====== 路徑設定 ======
WORKING_DIR="/kaggle/working/duplicate"
PREPROCESS_SCRIPT="$WORKING_DIR/preprocess_for_sagittal_classification.py"
TRAIN_SCRIPT="$WORKING_DIR/train_one_fold.py"
PREDICT_SCRIPT="$WORKING_DIR/predict.py"

# ====== 參數/環境變數 ======
configs=("$@")                           # 參數 = 要跑的 configs
read -r -a folds <<< "${FOLDS:-0}"       # 環境變數；可 "1" 或 "0 1"

if [ ${#configs[@]} -eq 0 ]; then
  echo "Error: no configs provided."; exit 2
fi

echo "Configs:"; printf '  - %s\n' "${configs[@]}"
echo "Folds:";   printf '  - %s\n' "${folds[@]}"

# ====== 可選：預處理 ======
# if [[ "${RUN_PREPROCESS:-0}" == "1" ]]; then
#   echo "Executing: python \"$PREPROCESS_SCRIPT\""
#   python "$PREPROCESS_SCRIPT"
# fi

# ====== 主流程 ======
for config in "${configs[@]}"; do
  for fold in "${folds[@]}"; do
    echo "Executing: python \"$TRAIN_SCRIPT\" -c \"$config\" -f \"$fold\""
    if ! python "$TRAIN_SCRIPT" -c "$config" -f "$fold"; then
      echo "Error: Training failed for config=$config fold=$fold."
      continue
    fi

    if [[ "${RUN_PREDICT:-0}" == "1" ]]; then
      echo "Executing: python \"$PREDICT_SCRIPT\" -c \"$config\" -f \"$fold\""
      python "$PREDICT_SCRIPT" -c "$config" -f "$fold" || \
        echo "Error: Prediction failed for config=$config fold=$fold."
    fi

    echo "----------------------------------------"
  done
done

echo "Script completed successfully!"
