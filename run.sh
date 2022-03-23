python train.py \
    --env BulletStack-v1
    --n_object 3
    --n_to_stack 1 2 3
    --num_train_steps 10000000
    --encoder_type identity \
    --save_video \
    --work_dir ./log_stack \
    --seed 1