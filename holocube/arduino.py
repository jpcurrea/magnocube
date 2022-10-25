# serial control of arduino

import serial
import numpy as n

class Arduino():
    '''Read and write to the arduino.'''

    def __init__(self):
        self.lmr_scale=.01
        self.lpr_scale=.02
        self.lpr_offset=10.
        self.hist = n.zeros((8,10000))
        self.hind = 0

    def start(self, serial_name='/dev/ttyACM0'):
        if serial_name=='dummy':
            self.ard = None
            self.read = self.dummy_read
            self.write = self.dummy_write
            self.waiting = self.dummy_waiting
        else:
            self.ard = serial.Serial(serial_name)
            self.read = self.serial_read
            self.write = self.serial_write
            self.waiting = self.serial_waiting

        
    def serial_read(self, channel=0):
       self.ard.write(chr(channel*2)) #request reading the channel
       return ord(self.ard.read(1))

    def dummy_read(self, channel=0):
        return n.random.randint(0,20)
    
    # def serial_write(self, channel=0, value=0):
    #     '''Write a high (1) or low (0) value to channels 0--7.'''
    #     self.ard.write(chr(channel*2 + 1 + 16*value))

    def serial_write(self, channel=0, value=0):
        '''Write pwm, channel 0--15, value 0--15'''
        self.ard.write(chr(packbits(unpackbits(array([1], dtype='ubyte')) +
                                    unpackbits(array([2*channel], dtype='ubyte')) +
                                    unpackbits(array([16*value], dtype='ubyte'))))[0])

    def write_channels(self, channels=[0], value=0):
        '''Write multiple pwm, channel 0--15, value 0--8'''
        for channel in channels:
            self.write(channel, value)

    def dummy_write(self, channel=0, value=0):
        '''Write a high (1) or low (0) value to channels 0--7.'''
        self.hist[channel, self.hind] = value
        self.hind = (self.hind + 1)%10000

    def serial_waiting(self):
        print (self.ard.inWaiting())

    def dummy_waiting(self):
        print (0, 'dummy')

    def pip(self, chan):
        self.write(chan, 1)
        self.write(chan, 0)
        
    def write0(self, value):
        self.write(0, value)

    def set_chan(self, chan):
        self.write_chan = chan

    def write_set_chan(self, value):
        self.write(self.write_chan, value)

    def set_lmr_scale(self, lmr_scale):
        self.lmr_scale = lmr_scale

    def set_lpr_scale(self, lpr_scale):
        self.lpr_scale = lpr_scale

    def set_lpr_offset(self, lpr_offset):
        self.lpr_offset = lpr_offset

    def lmr(self):
        return self.lmr_scale*(self.read(0) - self.read(1))

    def lpr(self):
        return self.lpr_scale*(self.read(0) + self.read(1)) + self.lpr_offset

    def led_flash(self, led_num):
        self.ard.write(bytes(chr(led_num), 'utf-8'))
