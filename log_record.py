import configure
import os


class LogRecoder:
    def __init__(self, resume):
        if os.path.exists(configure.ckpt_dir + 'log.txt') and not resume:
            raise RuntimeError
        else:
            try:
                os.makedirs(configure.ckpt_dir)
                print('Mkdir check point directory: {}.'.format(configure.ckpt_dir))
            except FileExistsError:
                # print('Check point directory exist. No need for mkdir.')
                pass

    def write(self, line):
        f = open(configure.ckpt_dir + 'log.txt', 'a')
        f.writelines(line + '\n')
        f.close()
