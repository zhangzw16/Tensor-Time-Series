import os

current_path = os.path.dirname(__file__)

'''
1. Script Header
params:
    - TASK_NAME
'''
Header=r'''
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 命令行传入 seed
seed=${1:-0}
task_name="<TASK_NAME>-${seed}"
'''
def make_header(task_name):
    header = Header.replace('<TASK_NAME>', task_name)
    return header

'''
2. Task Template
params:
    - task_idx
    - dataset
    - model
    - data Mode
    - his_len
    - pred_len
    - lag_input
    - batch_size
'''
TaskTemplate=r'''
# ====== Task<TASK_IDX> ======
dataset='<DATASET>'
model='<MODEL>'
data_mode=<DATA_MODE>
his_len=<HIS_LEN>
pred_len=<PRED_LEN>
lag_input=<LAG_INPUT>
batch_size=<BATCH_SIZE>
source $SCRIPT_DIR/run_task.sh
# ===================
'''
def make_task(task_idx, dataset, model, data_mode, his_len, pred_len, lag_input, batch_size):
    task = TaskTemplate.replace('<TASK_IDX>', str(task_idx))
    task = task.replace('<DATASET>', dataset)
    task = task.replace('<MODEL>', model)
    task = task.replace('<DATA_MODE>', str(data_mode))
    task = task.replace('<HIS_LEN>', str(his_len))
    task = task.replace('<PRED_LEN>', str(pred_len))
    task = task.replace('<LAG_INPUT>', str(lag_input))
    task = task.replace('<BATCH_SIZE>', str(batch_size))
    return task

def single_task():
    pass

def generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len_list, pred_len_list, lag_input, batch_size=0):
    save_path = os.path.join(current_path, f'{file_name}')
    # make header
    header = make_header(task_name)
    # task_list
    task_list = []
    task_count = 0
    for model in model_list:
        for dataset in dataset_list:
            for his_len in his_len_list:
                for pred_len in pred_len_list:
                    task = make_task(task_count, dataset, model, data_mode, his_len, pred_len, lag_input, batch_size)
                    task_list.append(task)
                    task_count += 1
    # save to file
    with open(save_path, 'w') as f:
        f.write(header)
        for task in task_list:
            task = '\n' + task
            f.write(task)
    print(f'Save to {save_path}')

if __name__ == '__main__':

    # =============================================================================================
    # lag input tasks -- Traffic
    his_len = [0]
    pred_len = [12]
    lag_input = '\'[4,7,12]\''
    data_mode = 0
    batch_size = 128
    task_name = 'Task2-Lag-Input'
    dataset_list = ['COVID_CHI', 'JONAS_NYC_taxi']
    # TTS-1
    model_list = ['GCGRU', 'MTGNN', 'AGCRN', 'GraphWaveNet']
    file_name = 'run_TTS-[GCGRU,MTGNN,AGCRN,GraphWaveNet].sh'
    generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len, pred_len, lag_input, batch_size)
    # TTS-2
    model_list = ['ST_Norm', 'NET3', 'Mamba', 'DCRNN']
    file_name = 'run_TTS-[ST_Norm,NET3,Mamba,DCRNN].sh'
    generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len, pred_len, lag_input, batch_size)
        
    # MTS-1
    model_list = ['DLinear', 'STID', 'TimesNet', 'PatchTST']
    file_name = 'run_MTS-[DLinear,STID,TimesNet,PatchTST].sh'
    generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len, pred_len, lag_input, batch_size)
    # MTS-2
    model_list = ['AutoFormer', 'CrossFormer', 'StemGNN', 'STWA']
    file_name = 'run_MTS-[AutoFormer,CrossFormer,StemGNN,STWA].sh'
    generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len, pred_len, lag_input, batch_size)
    # =============================================================================================
    