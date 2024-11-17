import time

class Timer:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.base_timestamp = time.time()
        self.timer_log = {}
    def mark_start_time(self, name:str):
        self.timer_log[name] = {
            'start': time.time() - self.base_timestamp,
            'end': 0,
            'duration': 0,
        }
    def mark_end_time(self, name:str):
        if name not in self.timer_log:
            raise KeyError(f"ID: {name} not in time log.")
        self.timer_log[name]['end'] = time.time() - self.base_timestamp
        self.timer_log[name]['duration'] = self.timer_log[name]['end'] - self.timer_log[name]['start']
        return self.timer_log[name]['duration']
    
    def save_timer_log(self, path='/home/zhuangjiaxin/workspace/TensorTSL/Tensor-Time-Series/output/timer_log.csv'):
        log_line = f"{self.model_name}, {self.dataset_name}"
        for name in self.timer_log:
            duration = self.timer_log[name]['duration']
            log_line += f", {duration:.3f}"
        log_line += '\n'
        file = open(path, '+a')
        file.write(log_line)
        file.close()