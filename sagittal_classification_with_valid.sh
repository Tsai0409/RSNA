# sagittal_classification.sh
#!/bin/bash
set -euo pipefail

WORKING_DIR="/kaggle/working/duplicate"
# PREPROCESS_SCRIPT="$WORKING_DIR/preprocess_for_sagittal_classification.py"
TRAIN_SCRIPT="$WORKING_DIR/train_one_fold.py"
# PREDICT_SCRIPT="$WORKING_DIR/predict.py"   # 需要推論再開

# 1) 先做一次預處理
# echo "Executing: python \"$PREPROCESS_SCRIPT\""
# python "$PREPROCESS_SCRIPT"

# 2) 參數：所有參數皆為要執行的 configs（可 1~10 個）
configs=("$@")
if [ ${#configs[@]} -eq 0 ]; then
  echo "Error: no configs provided. Pass 1-10 configs as arguments."
  exit 2
fi
if [ ${#configs[@]} -gt 10 ]; then
  echo "Warning: more than 10 configs provided; only the first 10 will be used."
  configs=("${configs[@]:0:10}")
fi

# 3) 單一 fold：從環境變數 FOLD 取得，未指定則預設 0
fold="${FOLD:-0}"

echo "Configs:"
printf '  - %s\n' "${configs[@]}"
echo "Fold: $fold"

# 4) 逐個 config 執行
for config in "${configs[@]}"; do
  echo "Executing: python \"$TRAIN_SCRIPT\" -c \"$config\" -f \"$fold\""
  if ! python "$TRAIN_SCRIPT" -c "$config" -f "$fold"; then
    echo "Error: Training failed for config=$config fold=$fold."
    continue
  fi

  # 如需推論就解開下面區塊
  # echo "Executing: python \"$PREDICT_SCRIPT\" -c \"$config\" -f \"$fold\""
  # python "$PREDICT_SCRIPT" -c "$config" -f "$fold" || \
  #   echo "Error: Prediction failed for config=$config fold=$fold."

  echo "----------------------------------------"
done

echo "Script completed successfully!"
