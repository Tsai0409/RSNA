# axial_classification.sh
#!/bin/bash

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"
PREPROCESS_SCRIPT="$WORKING_DIR/preprocess_for_axial_classification.py"
TRAIN_SCRIPT="$WORKING_DIR/train_one_fold.py"
PREDICT_SCRIPT="$WORKING_DIR/predict.py"

# 執行預處理 (finish)
# cmd="python $PREPROCESS_SCRIPT"
# echo "Executing: $cmd"
# if ! eval $cmd; then
#     echo "Error: Preprocessing failed."
#     exit 1
# fi

# 設置 configs 和 folds 變數
# configs=(
#     "rsna_axial_ss_nfn_x2_y2_center_pad0_with_valid"
#     "rsna_axial_ss_nfn_x2_y6_center_pad0_with_valid"
#     "rsna_axial_ss_nfn_x2_y8_center_pad10_with_valid"
    
#     "rsna_axial_spinal_dis3_crop_x05_y6_with_valid"
#     "rsna_axial_spinal_dis3_crop_x1_y2_with_valid"
# )
# folds=(0 1 2 3 4)
# folds=(0)

# 遍歷配置和摺疊數進行訓練與預測
# for config in "${configs[@]}"
# do
#     for fold in "${folds[@]}"
#     do
#         # 執行訓練腳本
#         cmd="python $TRAIN_SCRIPT -c $config -f $fold"
#         echo "Executing: $cmd"
#         if ! eval $cmd; then
#             echo "Error: Training failed for config $config fold $fold."
#             continue  # 跳過失敗的配置，繼續執行其他
#         fi

        # 執行預測腳本
        # infcmd="python $PREDICT_SCRIPT -c $config -f $fold"
        # echo "Executing: $infcmd"
        # if ! eval $infcmd; then
        #     echo "Error: Prediction failed for config $config fold $fold."
        #     continue  # 跳過失敗的配置，繼續執行其他
        # fi

#         echo "----------------------------------------"
#     done
# done

#!/bin/bash
set -euo pipefail

WORKING_DIR="/kaggle/working/duplicate"
TRAIN_SCRIPT="$WORKING_DIR/train_one_fold.py"
# PREDICT_SCRIPT="$WORKING_DIR/predict.py"   # 需要再開

# 用參數當 configs；用環境變數 FOLDS（空則預設 0）
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