import os
import yaml

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

class BestInputReader:
    def __init__(self, path:str):
        # best_input_path = os.path.join(current_path, 'best_input.yaml')
        self.best_input = yaml.safe_load(open(path, 'r'))
    def get_best_input(self, dataset:str, model_name:str, pred_len:list):
        pred_len_list = [12, 48, 96]
        input_len_list = self.best_input[model_name][dataset]
        idx = pred_len_list.index(pred_len)
        return input_len_list[idx]

class SctriptHelper:
    def __init__(self, task_name:str):
        header = make_header(task_name)
        self.header = header
        self.task_list = []
        self.task_idx = 0
    
    def add_task(self, dataset_list, model_list, data_mode, his_len_list, pred_len_list, batch_size=0):
        for model in model_list:
            for dataset in dataset_list:
                for his_len in his_len_list:
                    for pred_len in pred_len_list:
                        task = make_task(self.task_idx, dataset, model, data_mode, his_len, pred_len, batch_size)
                        self.task_list.append(task)
                        self.task_idx += 1
    
    def save_to_file(self, file_name:str):
        save_path = os.path.join(current_path, f'{file_name}')
        with open(save_path, 'w') as f:
            f.write(self.header)
            for task in self.task_list:
                task = '\n' + task
                f.write(task)
        print(f'Save to {save_path}')

if __name__ == '__main__':
    
    # =============================================================================================
    # Different Modes tasks -- Traffic, Energy, Finance, Weather
    best_input_path = os.path.join(current_path, 'best_input.yaml')
    # best_input = yaml.safe_load(open(best_input_path, 'r'))
    best_input_reader = BestInputReader(best_input_path)
    task_name = 'Task4-Different-Modes'
    pred_len = [12, 48, 96]
    dataset_list = ['crypto12', 'METRO_HZ', 'COVID_CHI', 'ETT_hour', 'weather', 'JONAS_NYC_taxi', 'stocknet']
    # ===============================
    # TTS - mode 0 and 1
    data_mode = 1
    pass


    # -------------------------------
    model_list = ['PatchTST', 'STID']
    # MTS -mode 0, 1, 2, 3
    script_helper = SctriptHelper(task_name)
    data_mode = 1
    batch_size = 0
    for model in model_list:
        for dataset_name in dataset_list:
            for pred in pred_len:
                input_len = best_input_reader.get_best_input(dataset_name, model, pred)
                script_helper.add_task([dataset_name], [model], data_mode, [input_len], [pred], batch_size)
    script_helper.save_to_file(f'run_MTS-[PatchTST,STID]-mode-{data_mode}.sh')
    # -------------------------------
    script_helper = SctriptHelper(task_name)
    data_mode = 2
    batch_size = 0
    for model in model_list:
        for dataset_name in dataset_list:
            for pred in pred_len:
                input_len = best_input_reader.get_best_input(dataset_name, model, pred)
                script_helper.add_task([dataset_name], [model], data_mode, [input_len], [pred], batch_size)
    script_helper.save_to_file(f'run_MTS-[PatchTST,STID]-mode-{data_mode}.sh')
    # -------------------------------
    script_helper = SctriptHelper(task_name)
    data_mode = 3
    batch_size = 0
    for model in model_list:
        for dataset_name in dataset_list:
            for pred in pred_len:
                input_len = best_input_reader.get_best_input(dataset_name, model, pred)
                script_helper.add_task([dataset_name], [model], data_mode, [input_len], [pred], batch_size)
    script_helper.save_to_file(f'run_MTS-[PatchTST,STID]-mode-{data_mode}.sh')
    

    # 'TimesNet', 'CrossFormer', 'DLinear'
    model_list = ['TimesNet', 'CrossFormer', 'DLinear']
    # MTS -mode 0, 1, 2, 3
    script_helper = SctriptHelper(task_name)
    data_mode = 1
    batch_size = 0
    for model in model_list:
        for dataset_name in dataset_list:
            for pred in pred_len:
                input_len = best_input_reader.get_best_input(dataset_name, model, pred)
                script_helper.add_task([dataset_name], [model], data_mode, [input_len], [pred], batch_size)
    script_helper.save_to_file(f'run_MTS-[TimesNet,CrossFormer,DLinear]-mode-{data_mode}.sh')
    # -------------------------------
    script_helper = SctriptHelper(task_name)
    data_mode = 2
    batch_size = 0
    for model in model_list:
        for dataset_name in dataset_list:
            for pred in pred_len:
                input_len = best_input_reader.get_best_input(dataset_name, model, pred)
                script_helper.add_task([dataset_name], [model], data_mode, [input_len], [pred], batch_size)
    script_helper.save_to_file(f'run_MTS-[TimesNet,CrossFormer,DLinear]-mode-{data_mode}.sh')
    # -------------------------------
    script_helper = SctriptHelper(task_name)
    data_mode = 3
    batch_size = 0
    for model in model_list:
        for dataset_name in dataset_list:
            for pred in pred_len:
                input_len = best_input_reader.get_best_input(dataset_name, model, pred)
                script_helper.add_task([dataset_name], [model], data_mode, [input_len], [pred], batch_size)
    script_helper.save_to_file(f'run_MTS-[TimesNet,CrossFormer,DLinear]-mode-{data_mode}.sh')