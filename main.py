from tasks.task_manager import TaskManager

if __name__=='__main__':
    # set default values
    output_dir = './output/'
    dataset_name = 'nasdaq100'
    model_name = 'NET3'

    # task manager
    manager = TaskManager(project_name='test',output_dir=output_dir)
    manager.TaskRun(dataset_name, model_name, only_test=False)
