```shell
python3 shell_run.py --his_len 12 --pred_len 12 --model_type Stat --dataset_type Traffic --batch_size 32 --output_dir path/to/save [ --model_name NET3 ]
```
model_type: [Stat, MultiVar, Tensor-prior, Tensor-learned, Tensor-none]
+ Tensor-prior: Tensor Model with prior graph
+ Tensor-leaned: Tensor Model with learned graph
+ Tensor-none: Tensor Model without graph

dataset_type: [Traffic, Natural, Energy]
+ batch size should be less than 10 if you select the Natural dataset.

Optional: --model_name: you can select a model to run
