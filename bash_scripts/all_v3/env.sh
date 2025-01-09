# shared arguments for all scripts
# ====== 需要修改的参数 ======
# [!] seed 是后续通过命令行参数传入
# export dataset_base='/data/Blob_EastUS/v-zhenwzhang/tensor_ts_datasets/Processed_Data/'
# export output_dir='/data/Blob_EastUS/v-zhenwzhang/log/tensor_ts_log/20241110/'
export dataset_base='/nas/datasets/Tensor-Time-Series-Dataset/Processed_Data'
export output_dir='/nas/datasets/zjx/datasets/TensorTSL_Output/lr_1e-4'
# ==== 任务配置 ====
export device='cuda'
export train_test='train'
export scheduler='ReduceLROnPlateau'
export logger='tensorboard'
export normalizer='sklearn'
export graph_init='pearson'
# export data_mode=0

