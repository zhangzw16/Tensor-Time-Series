import argparse
from tasks.task_manager import TaskManager

if __name__=='__main__':
    # set default values
    output_dir = '/home/zhuangjiaxin/workspace/TensorTSL/Tensor-Time-Series/output/'
    dataset_name = 'JONAS_NYC_taxi'
    model_name = 'TimesNet'

    # task manager
    manager = TaskManager(project_name='test',output_dir=output_dir)
    manager.TaskRun(dataset_name, model_name)
