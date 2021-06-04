import time
from enum import Enum


class PointType(Enum):
    """
    Enum for values of PointType column in the data list. Points with a value >= 0 are box points and have the index
    of the data point they belong to as value.
    """
    DATA = -1
    BORDER = -2


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

