def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def can_convert_to_float(value):
    try:
        float(value)
        return True
    except ValueError:
        print('WTF5')
        return False

import threading
import time

class RunWithTimeout(object):
    def __init__(self, function, args):
        self.function = function
        self.answer = None
        self.timeouted = False
        self.args = args

    def worker(self):
        self.answer = self.function(self.args)

    def run(self, timeout):
        thread = threading.Thread(target=self.worker)
        thread.start()
        thread.join(timeout)
        
        return self.answer