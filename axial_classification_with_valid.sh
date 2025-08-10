# axial_classification.sh
#!/bin/bash
set -euo pipefail

WORKING_DIR="/kaggle/working/duplicate"
TRAIN_SCRIPT="$WORKING_DIR/train_one_fold.py"
# PREDICT_SCRIPT="$WORKING_DIR/predict.py"   # 需要推論再開

# 參數當作要跑的 configs；FOLDS 為空則預設 0
configs=("$@")
read -r -a folds <<< "${FOLDS:-0}"

echo "Configs:"
printf '  - %s\n' "${configs[@]}"
echo "Folds:"
printf '  - %s\n' "${folds[@]}"

for config in "${configs[@]}"; do
  for fold in "${folds[@]}"; do
    echo "Executing: python \"$TRAIN_SCRIPT\" -c \"$config\" -f \"$fold\""
    if ! python "$TRAIN_SCRIPT" -c "$config" -f "$fold"; then
      echo "Error: Training failed for config=$config fold=$fold."
      continue
    fi
    # 如需推論，解除下列註解
    # echo "Executing: python \"$PREDICT_SCRIPT\" -c \"$config\" -f \"$fold\""
    # python "$PREDICT_SCRIPT" -c "$config" -f "$fold" || echo "Error: Prediction failed for config=$config fold=$fold."
    echo "----------------------------------------"
  done
done

echo "Script completed successfully!"
