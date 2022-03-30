import configure
import os


class LogRecoder:
    def __init__(self, resume):
        if os.path.exists(configure.ckpt_dir + 'log.txt') and not resume:
            raise RuntimeError
        else:
            pass

    def write(self, line):
        f = open(configure.ckpt_dir + 'log.txt', 'a')
        f.writelines(line + '\n')
        f.close()
