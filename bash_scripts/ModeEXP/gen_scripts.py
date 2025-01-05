import os
import yaml
import csv

current_path = os.path.dirname(__file__)

Header=r'''
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 命令行传入 seed
seed=${1:-0}
task_name="<TASK_NAME>-${seed}"
data_mode=<DATA_MODE>
'''

TaskTemplate=r'''
# ====== Task<TASK_IDX> ======
dataset='<DATASET>'
model='<MODEL>'
his_len=<HIS_LEN>
pred_len=<PRED_LEN>
batch_size=<BATCH_SIZE>
source $SCRIPT_DIR/run_task.sh
# ===================
'''

def make_header(task_name, data_mode):
    header = Header.replace('<TASK_NAME>', task_name)
    header = header.replace('<DATA_MODE>', data_mode)
    return header

def make_task(task_idx, dataset, model, his_len, pred_len, batch_size):
    task = TaskTemplate.replace('<TASK_IDX>', str(task_idx))
    task = task.replace('<DATASET>', dataset)
    task = task.replace('<MODEL>', model)
    task = task.replace('<HIS_LEN>', str(his_len))
    task = task.replace('<PRED_LEN>', str(pred_len))
    task = task.replace('<BATCH_SIZE>', str(batch_size))
    return task

def read_yaml(file_path):
    config = yaml.safe_load(open(file_path, 'r'))
    return config

if __name__=='__main__':
    TTS_1_models = ['NET3', 'MTGNN']
    TTS_2_models = ['AGCRN', 'GraphWaveNet']
    MTS_models = ['DLinear', 'STID', 'TimesNet', 'PatchTST']
    Datasets = ['JONAS_NYC_taxi', 'COVID_CHI', 'METRO_HZ', 'ETT_hour', 'weather', 'crypto12', 'stocknet']
    # 读取配置文件
    best_his = read_yaml(os.path.join(current_path, 'best_input.yaml'))
    # print(best_his['NET3']['JONAS_NYC_taxi'])
    task_name = 'ModeEXP-MTS'
    model_list = MTS_models
    data_mode = 3
    batch_size = 0  # enable AutoBatch
    SavePath = os.path.join(current_path, f'Mode-{data_mode}-MTS.sh')

    # header
    data_mode = str(data_mode)
    batch_size = str(batch_size)
    header = make_header(task_name, data_mode)
    # tasks
    task_list = []
    task_count = 0
    pred_len_list = [12, 48, 96]
    for model in model_list:
        for dataset in Datasets:
            his_len_list = best_his[model][dataset]
            for i in range(len(pred_len_list)):
                pred = str(pred_len_list[i])
                his = str(his_len_list[i])
                task_str = make_task(task_count, dataset, model, his, pred, batch_size)    
                task_list.append(task_str)
                task_count += 1
    # save
    with open(SavePath, 'w') as f:
        f.write(header)
        for task in task_list:
            task = '\n' + task
            f.write(task)
    print('Saved to:', SavePath)