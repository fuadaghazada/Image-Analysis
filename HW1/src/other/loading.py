import itertools
import threading
import time
import sys

'''
    For animation while waiting :D

    Reference: https://stackoverflow.com/questions/22029562/python-how-to-make-simple-animated-loading-while-process-is-running
'''
def execute(function):
    done = False

    def animate():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if done:
                sys.stdout.write('\n')
                break
            sys.stdout.write('\rPlease wait ' + c)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rFinished!\n')

    # Creating thread
    t = threading.Thread(target = animate)
    t.start()

    function()

    done = True
