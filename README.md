MagnoCube is a flight simulator for flies designed to have 1) a high resolution display, 2) magnetic tether, and 3) high-speed feedback used to update the display. To do this, the rig coordinates the following devices:
- Texas Instruments DLP Lightcrafter 4500 high speed projector 
- FLIR BlackFly S USB 3.0
- National Instruments DAQ 

To generate stimuli, we are using the Holocube program built by Jamie Theobald which uses OpenGL to quickly render 4 viewports, which are then projected onto the 4 vertical sides of a cube using the projector (goal #1). The fly is tethered at the center of the cube using a magnetic tether which allows the fly to rotate freely along the yaw axis (goal #2). The camera views the fly from below, allowing us to process its heading, for each frame ideally. This alone allows the fly to close the visuo-motor feedback loop themselves and is an excellent paradigm for psychophysical experiments. 
Finally, the rig relays the fly's heading to update the display (goal #3). This offers many useful features. For instance, we can now control the starting point of stimuli in closed-loop, such as always starting an experiment with a bar at the center of the visual field (or anywhere within the visual field, really). Previously, researchers had to rely on randomness to measure responses to stimuli in different locations. Moreso, this updating signal of the fly's heading allows us to remove visual feedback caused by the body's rotation. This allows for open-loop vision experiments (like the rigid tether paradigm) in the context of closed-loop proprioceptive feedback (unlike the rigid tether). 

To coordinate the camera, projector, and DAQ, we are developing a Python module with 3 major classes (Camera, Projector, and DAQ) that integrate well with Holocube.
