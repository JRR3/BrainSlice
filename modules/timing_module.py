import time as t_lib

class TimeIt():
    def __init__(self, name=None):
        self.name = name
        self.t0   = None
    def __enter__(self):
        self.t0 = t_lib.time()
    def __exit__(self, type_, value, traceback):
        if self.name:
            print('Timer for {:s}'.format(self.name))
        t = t_lib.time() - self.t0
        print('Elapsed time: {:0.3f} seconds'.format(t))
