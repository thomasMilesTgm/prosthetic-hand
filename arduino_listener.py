import serial
import numpy as np
from tempfile import TemporaryFile
import threading
import sys
from time import time
import os
import string

NUM_DIMS = 9
MAX_LENGTH = 20
KEYS = ['t', 'l', 'x', 'y', 'z', 'c0_', 'c1_', 'c2_', 'c3_']

data_dir = '/home/tmiles/data/prosthetic/dummy/'
port = '/dev/ttyACM0'

class Listener:

    def __init__(self, init_class=None):
        self.current_class = ord(init_class[0])

    def parseInput(self, raw_in):
        '''
        reads a packet
        :param raw_in: packed from arduino

        :return: usable data dictionary
        '''
        parsed = {}
        split = raw_in.split()
        if len(split) != NUM_DIMS:
            return None

        else:
            for channel in split:
                channel = str(channel)
                channel = channel.split("'")[1]     # Remove the byte crap that we get
                channel = channel.split("!")        # Split the data-type from the data
                parsed[channel[0]] = channel[1]     # TODO This errors sometimes

            for key in KEYS:
                try:
                    parsed[key]
                except (KeyError, IndexError):
                    # incomplete packet
                    return None

            parsed['l'] = self.current_class
            return parsed

    def listen(self):

        tnow = time()
        os.system('mkdir ' + data_dir + str(tnow))

        data = np.zeros((NUM_DIMS-1, MAX_LENGTH))# timestamp and all datapoints
        labels = np.zeros((2, MAX_LENGTH))       # timestamp and corresponding label
        try:
            ard = serial.Serial(port,115200,timeout=5)
        except SerialException:
            print("ERROR: SerialException")
            exit()

        print("Collecting data")

        tick = 0
        while tick < MAX_LENGTH:
            try:
                ard.reset_input_buffer()
                serial_in = ard.readline()
                parsed = self.parseInput(serial_in)
                if parsed != None:

                    # TODO, save here every time so crashing is ok
                    # add the label data
                    labels[0,tick] = parsed['t']    # timestamp
                    labels[1,tick] = parsed['l']    # label
                    # add the data itself
                    data[0, tick] = parsed['t']     # timestamp
                    data[1, tick] = parsed['x']     # x acceleration
                    data[2, tick] = parsed['y']     # y acceleration
                    data[3, tick] = parsed['z']     # z acceleration
                    data[4, tick] = parsed['c0_']   # EMG sensor 0
                    data[5, tick] = parsed['c1_']   # EMG sensor 1
                    data[6, tick] = parsed['c2_']   # EMG sensor 2
                    data[7, tick] = parsed['c3_']   # EMG sensor 3

                    np.save(data_dir + str(tnow) + '/data.npy', data)
                    np.save(data_dir + str(tnow) + '/labels.npy', labels)

                    tick = tick + 1

            except (KeyboardInterrupt):
                print(tick)
                ard.flushInput()
                pass


        print(data)






        print("MAX_LENGTH reached, saved to .npy files at: " + data_dir + str(tnow))
        exit()



    def update_label(self, L):
        self.current_class = L
        print("label is now: " + str(L))


# TODO THIS MAKES IT RECORD FOREVER -_-
def listen(ard_listener):

    print ('listener started')
    while True:

        # if not listenThread.is_alive():
        #     listenThread.run()

        L = sys.stdin.readline()

        ard_listener.update_label(L)



if __name__ == '__main__':
    threads = []

    ard_listener = Listener(init_class='r')
    threads.append(threading.Thread(target=ard_listener.listen))
    threads.append(threading.Thread(target=listen, args=(ard_listener,), daemon=True))

    for t in threads:
        t.start()








