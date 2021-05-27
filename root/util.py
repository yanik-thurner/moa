import time

class Task:
    """
    A helper class to measure execution time of certain tasks.
    """
    def __init__(self, task_name):
        self.name = task_name
        self.start_time = time.time()
        print(f'--- STARTING: {task_name} ---')

    def end(self):
        print(f'--- FINISHING:  {self.name} in {"{:.3f}".format(time.time() - self.start_time)}s')