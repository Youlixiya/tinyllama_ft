export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

python -m torch.distributed.run --nproc_per_node=8 \
         tinyllama_ft/train/finetune.py \
        --data_path data/sr3d_84k.json \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --bf16 \
        --output_dir checkpoints/tinyllama_ft \
        --max_steps 10000    \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 5000  \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --tf32 True  \
        --model_max_length 2048  \
        --gradient_checkpointing True  \
        --lazy_preprocess True \