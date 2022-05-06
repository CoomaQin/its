import os
import shutil
import numpy as np


class Spliter():
    def __init__(self, data_path='./data/train', output_path='./data/test'):
        self.data_path = data_path
        self.output_path = output_path

    def split_by_ratio(self, ratio=0.8):
        objs = []
        for (dirpath, dirnames, filenames) in os.walk(self.data_path):
            objs.extend(dirnames)
            break

        for obj in objs:
            r = np.random.rand()
            if r > ratio:
                shutil.copytree(f'{self.data_path}/{obj}', f'{self.output_path}/{obj}')  


if __name__ == "__main__":
    spliter = Spliter()
    spliter.split_by_ratio()