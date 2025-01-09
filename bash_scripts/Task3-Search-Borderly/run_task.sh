# 获取环境变量
python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
                    --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
                    --normalizer $normalizer --graph_init $graph_init \
                    --dataset_base $dataset_base \
                    --scheduler $scheduler \
                    --logger $logger \
                    --seed $seed \
                    --dataset $dataset --model $model
                    # --lr_finder