from camera import *

# Retrieve singleton reference to system object
system = PySpin.System.GetInstance()
# Get current library version
version = system.GetLibraryVersion()
print("PySpin version: "
      f"{version.major}.{version.minor}.{version.type}.{version.build}")
# Retrieve list of cameras from the system
cam_list = system.GetCameras()
num_cameras = cam_list.GetSize()
print('Number of cameras detected:', num_cameras)
# Finish if there are no cameras
if num_cameras > 0:
    # test: profile using the 3 different capture functions
    cam = Camera(cam_list[0])
    for capture_func in ['capture_np', 'capture_l', 'capture_q']:
        # Use the first indexed camera
        setattr(cam, 'capture', getattr(cam, capture_func))
        cam.grab_frames(10)
        # start profiler
        profiler = cProfile.Profile()
        profiler.enable()
        cam.capture_start(duration=4, save_fn="test.mp4")
        # stop profiler
        profiler.disable()
        # save the benchmark data
        txt_fn = capture_func + '.txt'
        profiler.dump_stats(txt_fn)
        # get pstats instance and print the top 10 longest duration entries
        benchmark = pstats.Stats(txt_fn)
        benchmark.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
        # Release reference to camera
        cam.save_thread.join()
    del cam
else:
    print('No cameras!')
# Clear camera list before releasing system
cam_list.Clear()
# Release instance
system.ReleaseInstance()
