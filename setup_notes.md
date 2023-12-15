Here are instructions for setting up a new computer. Refer to a requirements.txt for all of the typical Python dependencies. This may eventually develop into a setup.py script to automate this setup, but for now, since I'm still making changes to holocube, I'll just make a list. 

Manually install the following for accessing the zotero library and running typical processes:
0. Copy the two registry files from the Drive (H) Desktop folder to swap ctrl and caps lock (caps_to_ctrl.reg) and to allow long file paths(enable_max_path). 
1. Firefox and Thunderbird with 2 main email accounts: pablocurrea@g.ucla.edu and johnpaulcurrea@gmail.com
2. Google drive and the two accounts. Set them to stream and then select the key folders you want to keep local copies.
3. Zotero

For setting up a magnocube:
1. FLIR software:
    - install SpinView and keep track of the version number
2. ffmpeg:
    - install the latest version of ffmpeg at ffmpeg.org
    - remember to add the ffmpeg/bin folder to the path environment variable. To check this, you should be able to run the "ffmpeg" command from the command prompt. 
3. Python3.10+
    - install a python 3.10 or greater
    - add the Scripts folder to the path environment variable
        - on my windows machine, this is found under C:\Users\{username}\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts
4. holocube:
    - git clone the file and install from requirements.txt using pip:
    ```pip install -r requirements.txt```
    - pyglet is the only package that requires a specific version, but requirements.txt should specify that. You can manually install the last working verion with pip:
    ```pip install pyglet==1.5.27```
5. try it!
    - if everything installed as expected, you should be able to run the test.py and run.py scripts. 
    - test.py makes a window in the main display to help test new stimuli
    - run.py uses a secondary display to project stimuli on a cube arena, for instance

todo:
-----
DAQ: https://www.ni.com/docs/en-US/bundle/daq-getting-started-bus-powered-usb/page/getting-started.html
    - the goal of the DAQ is to keep track of the exact time when the camera and the projector each trigger a frame. So the program will use the DAQ to measure 2 time series with the exact time when it receives a projector or camera trigger. This will be saved as another dataset in the .h5 file.