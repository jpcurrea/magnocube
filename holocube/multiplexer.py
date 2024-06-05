"""Tools for interacting with the Sable Systems Intelligent Multiplexer

Here is an object for easily interacting with the multiplexer within
the holocube environment. It's easy to set the active channel of the multiplexer
using the serial connection. The connection requires a special USB to Serial RS232
Adapter Cable with COM retention (StarTech.com), which comes with a downloadable 
driver. Once installed, you can communicate using the pyserial library by setting 
the device to to Serial mode (looks like "SERIAL 000 - 255"). Then, it will interpret
bytestrings of the following form:

"mmux:{1-255}\n"

The number is determined by the binary encoding of the desired channels. It 
is effectively the decimal form of the binary code where the first digit corresponds
to channel 1, the second channel 2, and so on:

num = 1 * (channel_1) + 2 * (channel_2) + 4 * (channel_3) + 8 * (channel_4)

where each channel variable is a boolean (1 means ON)

The object here establishes the serial port connection and allows for easy 
conversion to the binary values.
"""
import serial

class Multiplexer:
    def __init__(self, port='COM3', baudrate=9600):
        """Initialize the multiplexer object"""
        self.port = port
        self.baudrate = baudrate
        self.channels = [False] * 8
        self.num = 0
        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate)

    def set_channel(self, channel, state):
        """Set the state of a single channel on the multiplexer"""
        self.channels[channel] = state

    def set_channels(self, channels):
        """Set the state of all channels on the multiplexer by passing a list of booleans"""
        self.channels = channels
        self.write()

    def all_off(self):
        """Turn all channels off"""
        self.channels = [False] * 8
        self.write()

    def write(self):
        """Write the current state of the channels to the multiplexer"""
        self.num = sum([2**i * self.channels[i] for i in range(8)])
        self.ser.write(f"mmux:{self.num}\n".encode())
        # print(f"mmux:{self.num}\n")

    def close(self):
        self.ser.close()
        print('Multiplexer closed')