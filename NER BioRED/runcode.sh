#!/bin/bash
python3 run_ner.py \
    --data_dir='' \
    --model_name_or_path='' \
    --output_dir='' \
    --max_seq_length=512 \
    --num_train_epochs=50 \
    --per_device_train_batch_size=10 \
    --logging_steps=100 \
    --seed=123 \
    --metric_for_best_model=eval_f1 \
    --learning_rate=5e-5 \
    --weight_decay=0.01 \
    --warmup_steps=0 \
    --do_train=True \
    --do_eval=True \
    --do_predict=False \
    --overwrite_output_dir=False