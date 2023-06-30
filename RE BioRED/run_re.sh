#!/bin/bash
python3 run_re2.py \
    --data_dir='' \
    --model_name_or_path='' \
    --output_dir='' \
    --tokenizer_name='' \
    --max_seq_length=512 \
    --num_train_epochs=10 \
    --per_device_train_batch_size=12 \
    --logging_steps=100 \
    --seed=123 \
    --dataloader_num_workers=4 \
    --metric_for_best_model=eval_f1_micro \
    --learning_rate=5e-4 \
    --weight_decay=0.01 \
    --warmup_steps=0 \
    --do_train=True \
    --do_eval=True \
    --do_predict=False \
    --overwrite_output_dir=False