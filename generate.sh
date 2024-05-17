#!/bin/bash

# Check if the required argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_path>"
    exit 1
fi

# Assign the first argument to the output_path variable
output_path="$1"

# Create a new tmux session named "mysession"
tmux new-session -d -s generation


# Define the list of commands
commands=(
    "CUDA_VISIBLE_DEVICES=0 python 4_StableDiffusion.py --coco_path ./coco --modified_coco_path ./modified_coco --mit_ckpt_path ./checkpoints/train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler_0.1_0.7_aug30/epoch=337-Cls_Acc=64.597.ckpt --mgan_ckpt_path ./checkpoints/train_mgan_default_diff_aug_true_attn_false_extra_feature_condition_false_batch_256_0.1_0.7_aug30/epoch=191-val_fid=15.675.ckpt --bsc_ckpt_path ./checkpoints/train_bsc_vanilla_0.1_0.7_aug30/epoch=79-valid_mIoU=0.603.ckpt --output_path \"$output_path\" --start 0.0 --end 0.125"
    "CUDA_VISIBLE_DEVICES=1 python 4_StableDiffusion.py --coco_path ./coco --modified_coco_path ./modified_coco --mit_ckpt_path ./checkpoints/train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler_0.1_0.7_aug30/epoch=337-Cls_Acc=64.597.ckpt --mgan_ckpt_path ./checkpoints/train_mgan_default_diff_aug_true_attn_false_extra_feature_condition_false_batch_256_0.1_0.7_aug30/epoch=191-val_fid=15.675.ckpt --bsc_ckpt_path ./checkpoints/train_bsc_vanilla_0.1_0.7_aug30/epoch=79-valid_mIoU=0.603.ckpt --output_path \"$output_path\" --start 0.125 --end 0.25"
    "CUDA_VISIBLE_DEVICES=2 python 4_StableDiffusion.py --coco_path ./coco --modified_coco_path ./modified_coco --mit_ckpt_path ./checkpoints/train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler_0.1_0.7_aug30/epoch=337-Cls_Acc=64.597.ckpt --mgan_ckpt_path ./checkpoints/train_mgan_default_diff_aug_true_attn_false_extra_feature_condition_false_batch_256_0.1_0.7_aug30/epoch=191-val_fid=15.675.ckpt --bsc_ckpt_path ./checkpoints/train_bsc_vanilla_0.1_0.7_aug30/epoch=79-valid_mIoU=0.603.ckpt --output_path \"$output_path\" --start 0.25 --end 0.375"
    "CUDA_VISIBLE_DEVICES=3 python 4_StableDiffusion.py --coco_path ./coco --modified_coco_path ./modified_coco --mit_ckpt_path ./checkpoints/train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler_0.1_0.7_aug30/epoch=337-Cls_Acc=64.597.ckpt --mgan_ckpt_path ./checkpoints/train_mgan_default_diff_aug_true_attn_false_extra_feature_condition_false_batch_256_0.1_0.7_aug30/epoch=191-val_fid=15.675.ckpt --bsc_ckpt_path ./checkpoints/train_bsc_vanilla_0.1_0.7_aug30/epoch=79-valid_mIoU=0.603.ckpt --output_path \"$output_path\" --start 0.375 --end 0.5"
    "CUDA_VISIBLE_DEVICES=4 python 4_StableDiffusion.py --coco_path ./coco --modified_coco_path ./modified_coco --mit_ckpt_path ./checkpoints/train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler_0.1_0.7_aug30/epoch=337-Cls_Acc=64.597.ckpt --mgan_ckpt_path ./checkpoints/train_mgan_default_diff_aug_true_attn_false_extra_feature_condition_false_batch_256_0.1_0.7_aug30/epoch=191-val_fid=15.675.ckpt --bsc_ckpt_path ./checkpoints/train_bsc_vanilla_0.1_0.7_aug30/epoch=79-valid_mIoU=0.603.ckpt --output_path \"$output_path\" --start 0.5 --end 0.625"
    "CUDA_VISIBLE_DEVICES=5 python 4_StableDiffusion.py --coco_path ./coco --modified_coco_path ./modified_coco --mit_ckpt_path ./checkpoints/train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler_0.1_0.7_aug30/epoch=337-Cls_Acc=64.597.ckpt --mgan_ckpt_path ./checkpoints/train_mgan_default_diff_aug_true_attn_false_extra_feature_condition_false_batch_256_0.1_0.7_aug30/epoch=191-val_fid=15.675.ckpt --bsc_ckpt_path ./checkpoints/train_bsc_vanilla_0.1_0.7_aug30/epoch=79-valid_mIoU=0.603.ckpt --output_path \"$output_path\" --start 0.625 --end 0.75"
    "CUDA_VISIBLE_DEVICES=6 python 4_StableDiffusion.py --coco_path ./coco --modified_coco_path ./modified_coco --mit_ckpt_path ./checkpoints/train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler_0.1_0.7_aug30/epoch=337-Cls_Acc=64.597.ckpt --mgan_ckpt_path ./checkpoints/train_mgan_default_diff_aug_true_attn_false_extra_feature_condition_false_batch_256_0.1_0.7_aug30/epoch=191-val_fid=15.675.ckpt --bsc_ckpt_path ./checkpoints/train_bsc_vanilla_0.1_0.7_aug30/epoch=79-valid_mIoU=0.603.ckpt --output_path \"$output_path\" --start 0.75 --end 0.875"
    "CUDA_VISIBLE_DEVICES=7 python 4_StableDiffusion.py --coco_path ./coco --modified_coco_path ./modified_coco --mit_ckpt_path ./checkpoints/train_mit_type_1_abs4c_no_classifier_lr_1.0e-4_deep_scheduler_0.1_0.7_aug30/epoch=337-Cls_Acc=64.597.ckpt --mgan_ckpt_path ./checkpoints/train_mgan_default_diff_aug_true_attn_false_extra_feature_condition_false_batch_256_0.1_0.7_aug30/epoch=191-val_fid=15.675.ckpt --bsc_ckpt_path ./checkpoints/train_bsc_vanilla_0.1_0.7_aug30/epoch=79-valid_mIoU=0.603.ckpt --output_path \"$output_path\" --start 0.875 --end 1.0"
)

# Create additional tmux windows and send commands to them
window_index=3
for cmd in "${commands[@]}"; do
    # Create a new tmux window for each command
    tmux new-window -t generation:$window_index -n "Generation $((window_index-2))"
    # Send the command to the window
    tmux send-keys -t generation:$window_index "$cmd" C-m
    window_index=$((window_index + 1))
done

# Attach to the tmux session to see the progress
# tmux attach -t mysession
