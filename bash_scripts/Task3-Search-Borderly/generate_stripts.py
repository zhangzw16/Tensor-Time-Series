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
    - batch_size
'''
TaskTemplate=r'''
# ====== Task<TASK_IDX> ======
dataset='<DATASET>'
model='<MODEL>'
data_mode=<DATA_MODE>
his_len=<HIS_LEN>
pred_len=<PRED_LEN>
batch_size=<BATCH_SIZE>
source $SCRIPT_DIR/run_task.sh
# ===================
'''
def make_task(task_idx, dataset, model, data_mode, his_len, pred_len, batch_size):
    task = TaskTemplate.replace('<TASK_IDX>', str(task_idx))
    task = task.replace('<DATASET>', dataset)
    task = task.replace('<MODEL>', model)
    task = task.replace('<DATA_MODE>', str(data_mode))
    task = task.replace('<HIS_LEN>', str(his_len))
    task = task.replace('<PRED_LEN>', str(pred_len))
    task = task.replace('<BATCH_SIZE>', str(batch_size))
    return task

def single_task():
    pass

def generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len_list, pred_len_list, batch_size=0):
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
                    task = make_task(task_count, dataset, model, data_mode, his_len, pred_len, batch_size)
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
    # search borderly tasks -- Traffic, Energy, Finance, Weather
    his_len = [1, 3, 6, 9]
    pred_len = [12, 48, 96]
    data_mode = 0
    task_name = 'Task3-Search-Borderly'
    dataset_list = ['crypto12', 'COVID_CHI']
    # TTS-1
    model_list = ['GraphWaveNet', 'AGCRN']
    file_name = 'run_TTS-[GraphWaveNet,AGCRN].sh'
    generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len, pred_len)
    # TTS-2
    model_list = ['GCGRU', 'MTGNN']
    file_name = 'run_TTS-[GCGRU,MTGNN].sh'
    generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len, pred_len)
    # MTS
    model_list = ['DLinear', 'TimesNet', 'PatchTST', 'STID']
    file_name = 'run_MTS-[DLinear,TimesNet,PatchTST,STID].sh'
    generate_stripts(file_name, task_name, dataset_list, model_list, data_mode, his_len, pred_len)
    