import configure
import os


class LogRecoder:
    def __init__(self):
        if os.path.exists(configure.model_dir + '/torch/log.txt'):
            raise RuntimeError
        else:
            pass

    def write(self, line):
        f = open(configure.model_dir + '/torch/log.txt', 'a')
        f.writelines(line + '\n')
        f.close()
