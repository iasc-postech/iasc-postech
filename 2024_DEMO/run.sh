CUDA_VISIBLE_DEVICES=0 python demo.py \
    --root_folder /root/data/KETI_VISUAL_COMMONSENSE_2023/KETI_SG_DB/val \
    --output_folder ./val_generated_refine_2 \
    --refine_type 2 

CUDA_VISIBLE_DEVICES=0 python demo.py \
    --root_folder /root/data/KETI_VISUAL_COMMONSENSE_2023/KETI_SG_DB/test \
    --output_folder ./test_generated_refine_2 \
    --refine_type 2

CUDA_VISIBLE_DEVICES=0 python demo.py \
    --root_folder /root/data/KETI_VISUAL_COMMONSENSE_2023/KETI_SG_DB/val \
    --output_folder ./val_generated_refine_1 \
    --refine_type 1 \

CUDA_VISIBLE_DEVICES=0 python demo.py \
    --root_folder /root/data/KETI_VISUAL_COMMONSENSE_2023/KETI_SG_DB/test \
    --output_folder ./test_generated_refine_1 \
    --refine_type 1
