# windowing classes

import pyglet
from pyglet.gl import *
from pyglet.window import key
from numpy import *
import numpy as np
from os.path import expanduser
import time
import configparser

# for rotating the camera view angles
def rotmat(u=[0.,0.,1.], theta=0.0):
    '''Returns a matrix for rotating an arbitrary amount (theta) around an
    arbitrary axis (u, a unit vector).
    '''
    ux, uy, uz = u
    cost, sint = cos(theta), sin(theta) # tensor product u x u
    u_tp = array([[ux*ux, ux*uy, ux*uz],
                  [ux*uy, uy*uy, uz*uy],
                  [ux*uz ,uy*uz , uz*uz]]) 
    u_cpm = array([[0, -uz, uy],        # u cross product matrix u_x
                   [uz, 0, -ux],
                   [-uy, ux, 0]])
    return cost*identity(3) + sint*u_cpm + (1 - cost)*u_tp


class Viewport_class():
    '''A class for a single window configuration information.'''
    def __init__(self, batch, name='vp', coords=[0,0,150,150], scale_factors=[1.,1.,1],
                 for_ax=[2,-1], up_ax=[1,1], projection='perspective',
                 frustum=[-.1,.1,-.1,.1,.1,2.0], ref_pt_size=5,
                 ref_coords=[16.0,80.0,0.0,82.0,82.0,0.0,14.0,16.0,0.0,83.0,16.0,0.0],
                 pan=0.0, tilt=0.0, dutch=0.0):
        self.batch = batch
        self.name = name
        self.coords = array(coords, dtype='int') # window left, bottom, width, height
        self.orth_coords = self.coords
        self.scale_factors = scale_factors       # x y and z flips for mirror reflections
        self.pan = pan
        self.tilt = tilt
        self.dutch = dutch
        # calculate the intrinsic camera rotation matrix
        self.set_cam_rot_mat()
        # self.rmat = dot(rotmat([0.,1.,0.], azimuth), rotmat([1.,0.,0.], elevation))
        self.projection = projection             # orthographic or perspective projection, or ref
        # self.rotmat = dot(rotmat([0., 1., 0.], pan*pi/180), rotmat([1., 0., 0.], tilt*pi/180))
        self.forward_up = array([[0,0], [0,1], [-1,0.]])
        self.num_ref_pts = 0
        self.ref_coords = array([])
        self.ref_colors = array([])
        self.ref_vl = None
        if projection.startswith('ortho'):
            self.project = glOrtho
            self.draw = self.draw_ortho
            self.orth_coords =  self.coords
        elif projection.startswith('persp') :
            self.project = glFrustum
            self.draw = self.draw_perspective
        elif projection.startswith('ref'):
            self.ref_pt_size = ref_pt_size
            self.ref_coords = ref_coords
            self.num_refs = len(self.ref_coords)//3
            self.ref_colors = tile((0,0,0),(self.num_refs,1)).flatten()
            self.draw = self.draw_ref
            self.project = glOrtho
            self.batch = pyglet.graphics.Batch()
            self.ref_vl = self.batch.add(self.num_refs, GL_POINTS, None,
                                     ('v3f/static', self.ref_coords.flatten()),
                                     ('c3B/stream', self.ref_colors.flatten()))
            self.color = []
            for pt in range(self.num_refs):
                self.color.append(self.ref_vl.colors[(pt*3):(pt*3+3)])
            self.flash_patterns = {}
        self.frustum = frustum #left right bottom top near far
        self.ortho = frustum
        self.sprite_pos = 0
        self.clear_color = [0.,0.,0.,1.]

    def set_cam_rot_mat(self):
        '''Set the rotation matrix for the viewport from the pan, tilt, and
        dutch heading of the virtual camera, set as intrinsic fields
        before calling this function
        '''
        sint, cost = sin(self.pan*pi/180), cos(self.pan*pi/180)
        pan_mat = array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]]) # yaw of the camera
        sint, cost = sin(self.tilt*pi/180), cos(self.tilt*pi/180)
        tilt_mat = array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])# pitch of the camera
        sint, cost = sin(self.dutch*pi/180), cos(self.dutch*pi/180)
        dutch_mat = array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]]) # dutch (roll) of the camera
        self.cam_rot_mat = dot(dot(dutch_mat, tilt_mat), pan_mat)
        
    def draw_perspective(self, pos, rot):
        '''Clear the boundaries of the viewport with the background color,
        apply the scale factors (for any stretches or mirror
        reflections), then turn the camera with the rotation matrix,
        then draw all the
        '''
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(*self.frustum)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # erase the window
        glClearColor(*self.clear_color)
        glScissor(*self.coords)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # set the new viewport
        glViewport(*self.coords)
        # flip it if any mirror projections are required
        glScalef(*self.scale_factors)
        # point the camera
        view = dot(dot(rot, self.cam_rot_mat), self.forward_up)
        gluLookAt(pos[0],
                  pos[1],
                  pos[2],
                  pos[0] + view[0,0],
                  pos[1] + view[1,0],
                  pos[2] + view[2,0],
                  view[0,1],
                  view[1,1],
                  view[2,1] )
        # draw everything
        self.batch.draw()


    def draw_ortho(self, pos, rot, reflect=False):
        # I have to set reflect to false for sprites since sprites don't scale to
        # negative values right now, and the main use of ortho is to display sprites
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # gluOrtho2D(self.sprite_pos, self.sprite_pos + self.coords[2], 0, self.coords[3])
        # if not self.half: gluOrtho2D(0, self.coords[2], 0, self.coords[3])
        # else: gluOrtho2D(0, self.coords[2]/2, 0, self.coords[3]/2)
        gluOrtho2D(0, self.orth_coords[2], 0, self.orth_coords[3])
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # erase the window
        glClearColor(*self.clear_color)
        glScissor(*self.coords)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # set the new viewport
        glViewport(*self.coords)
        # flip it if any mirror projections are required
        if not reflect:
            glScalef(*[abs(sf) for sf in self.scale_factors])
        else: #
            glScalef(*self.scale_factors)
        # point the camera --- ignore pos and ori for ortho projection
        gluLookAt (self.sprite_pos, 0.0, 0.0, self.sprite_pos, 0.0, -100.0, 0.0, 1.0, 0.0)
        # draw everything
        self.batch.draw()

    def draw_ref(self, pos, rot):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0,100,0,100)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # erase the window
        glClearColor(*self.clear_color)
        glScissor(*self.coords)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # set the new viewport
        glViewport(*self.coords)
        # flip it if any mirror projections are required
        glScalef(*self.scale_factors)
        # point the camera
        glPointSize(10)
        gluLookAt(0,0,0,0,0,-1,0,1,0)
        # gluLookAt(0,0,0,50,50,-1,0,1,0)
        # draw everything
        self.batch.draw()

    def set_val(self, field, val, increment=False):
        '''Set any of the viewport positional and rotational parameters by
        name'''
        if   field=='left':   self.coords[0] = self.coords[0]*increment + val
        elif field=='bottom': self.coords[1] = self.coords[1]*increment + val
        elif field=='width':  self.coords[2] = self.coords[2]*increment + val
        elif field=='height': self.coords[3] = self.coords[3]*increment + val
        elif field=='near':   self.frustum[4] = self.frustum[4]*increment + val
        elif field=='far':    self.frustum[5] = self.frustum[5]*increment + val
        elif field=='scalex': self.scale_factors[0] *= -1
        elif field=='scaley': self.scale_factors[1] *= -1
        elif field=='scalez': self.scale_factors[2] *= -1
        elif field=='pan'   :
            self.pan = self.pan*increment + val
            self.set_cam_rot_mat()
        elif field=='tilt'   :
            self.tilt = self.tilt*increment + val
            self.set_cam_rot_mat()
        elif field=='dutch'   :
            self.dutch = self.dutch*increment + val
            self.set_cam_rot_mat()
        elif field=='bg':
            self.clear_color[0] = self.clear_color[0]*increment + val[0]
            self.clear_color[1] = self.clear_color[1]*increment + val[1]
            self.clear_color[2] = self.clear_color[2]*increment + val[2]
            self.clear_color[3] = self.clear_color[3]*increment + val[3]
        

        
class Holocube_window(pyglet.window.Window):
    '''Subclass of pyglet's window, for displaying all the viewports'''
    def __init__(self, ):
        # init the window class
        super(Holocube_window, self).__init__(
            style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS,
            # style=pyglet.window.Window.WINDOW_STYLE_DEFAULT,
            vsync=1, )

        self.disp = pyglet.canvas.get_display()    #**** New ****
        self.screens = self.disp.get_screens()
        self.w, self.h = self.get_size()

        self.moves = []
        self.key_actions = {}        #keys that execute a command, once, multiple times, or indefinitely
        self.frame_actions = []      #additional functions to execute each frame
        self.params = [[],[],[],[]]
        self.activate() # give keyboard focus

        # make the batches for display
        self.world = pyglet.graphics.Batch()

        # where's the camera?
        self.pos = zeros((3))
        self.rot = identity(3) # rotation matrix
        self.frame = 0
        self.add_keypress_action(key.I, self.print_info)
        self.bufm = pyglet.image.get_buffer_manager()

        # new hisory structure, to be declared at the beginning of an experiment
        self.hist_num_frames = 0
        self.hist_num_tests = 0
        self.hist_test_ind = 0
        self.record_fn_ind = 0          #current file number appended
        self.record_data = array([0])

        self.bg_color = [0.,0.,0.,1.]

        glEnable(GL_DEPTH_TEST)	# Enables Depth Testing
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_SCISSOR_TEST)
        # glPointSize(3)

        # store important parameters for the record
        self.yaw = 0
        self.record_params = []
        self.record = {}
        self.record_ind = 0
        # to improve closing behavior
        self.alive = 1

    def start(self, config_file='viewport.config'):
        '''Instantiate the window by loading the config file'''
        self.viewports = []
        self.load_config(config_file)
        # set the background color
        self.set_bg(self.bg_color)
        if self.project:
            self.set_fullscreen(True, self.screens[self.screen_number])
        else:
            self.set_size(self.w, self.h)
            # self.set_location(0, 400)
        print (self.screen_number, self.project)
        self.curr_viewport_ind = 0
        self.curr_indicator_ind = 0

    def load_config(self, filename='viewport.config'):
        '''Read the configuration file to size and color the window and all
        the viewports
        '''
        config = configparser.ConfigParser()
        config.read(filename)
        # grab options for the whole screen
        try:
            self.bg_color = eval(config.get('screen', 'bg_color', fallback=['[0.,0.,0.,1.]']))
        except:
            breakpoint()
        self.project = config.getboolean('screen', 'project', fallback=False)
        self.screen_number = config.getint('screen', 'screen_number', fallback=0)
        self.w = config.getint('screen', 'w_size', fallback=640)
        self.h = config.getint('screen', 'h_size', fallback=320)
        # remove the screen section if it's there
        if 'screen' in config.sections(): screen = config.pop('screen')
        # cycle through the remaining sections, viewports
        for vp_name in config.sections():
            batch         = self.world
            cfg           = config[vp_name]
            name          = vp_name
            coords        = [cfg.getint('left', 0), cfg.getint('bottom', 0), cfg.getint('width', 100), cfg.getint('height', 100)]
            scale_factors = [cfg.getfloat('scale_x', 1), cfg.getfloat('scale_y', 1), cfg.getfloat('scale_z', 1)]
            pan           = cfg.getfloat('pan', 0.0)
            tilt          = cfg.getfloat('tilt', 0.0)
            dutch         = cfg.getfloat('dutch', 0.0)
            projection    = cfg.get('projection', 'perspective')
            frustum       = [cfg.getfloat('frustum_left', -.1), cfg.getfloat('frustum_right', .1), cfg.getfloat('frustum_bottom', -.1), cfg.getfloat('frustum_top', .1), cfg.getfloat('frustum_near', .1), cfg.getfloat('frustum_far', 1.)]
            ref_pt_size   = cfg.getint('ref_pt_size', 1)
            ref_coords    = array([float(f) for f in cfg.get('ref_coords', '0. 0. 0.').split(' ')])
            if projection.startswith('ref'):
                batch = pyglet.graphics.Batch()
            self.add_viewport(batch, name, coords, scale_factors,
                              projection=projection, frustum=frustum,
                              ref_pt_size=ref_pt_size, ref_coords=ref_coords,
                              pan=pan, tilt=tilt, dutch=dutch)
 
    def save_config(self, filename='viewport.config'):
        '''Save the configuration file'''
        config = configparser.ConfigParser()
        config.add_section('screen')
        config['screen']['bg_color']                = str(self.bg_color)
        config['screen']['project']                 = str(self.project)
        config['screen']['screen_number']           = str(self.screen_number)
        config['screen']['w_size']                  = str(self.w)
        config['screen']['h_size']                  = str(self.h)
        for viewport in self.viewports:
            config.add_section(viewport.name)
            config[viewport.name]['left']           = str(viewport.coords[0])
            config[viewport.name]['bottom']         = str(viewport.coords[1])
            config[viewport.name]['width']          = str(viewport.coords[2])
            config[viewport.name]['height']         = str(viewport.coords[3])
            config[viewport.name]['scale_x']        = str(viewport.scale_factors[0])
            config[viewport.name]['scale_y']        = str(viewport.scale_factors[1])
            config[viewport.name]['scale_z']        = str(viewport.scale_factors[2])
            config[viewport.name]['projection']     = str(viewport.projection)
            if viewport.projection.startswith('persp'): # perspective projection
                config[viewport.name]['pan']            = str(viewport.pan)
                config[viewport.name]['tilt']           = str(viewport.tilt)
                config[viewport.name]['dutch']          = str(viewport.dutch)
                config[viewport.name]['frustum_left']   = str(viewport.frustum[0])
                config[viewport.name]['frustum_right']  = str(viewport.frustum[1])
                config[viewport.name]['frustum_bottom'] = str(viewport.frustum[2])
                config[viewport.name]['frustum_top']    = str(viewport.frustum[3])
                config[viewport.name]['frustum_near']   = str(viewport.frustum[4])
                config[viewport.name]['frustum_far']    = str(viewport.frustum[5])
            elif viewport.projection.startswith('ref'): # reference projection
                config[viewport.name]['ref_pt_size']    = str(viewport.ref_pt_size)
                config[viewport.name]['ref_coords']     = ' '.join([str(rc) for rc in viewport.ref_coords])
        # save the file
        with open(filename, 'w') as config_file:
            config.write(config_file)
        print('wrote {}'.format(config_file))

    def viewport_inc_ind(self, val=1, highlight=True):
        # get rid of old highlights
        if highlight: self.viewports[self.curr_viewport_ind].clear_color = [0.0, 0.0, 0.0, 1.0]
        # switch the index
        self.curr_viewport_ind = mod(self.curr_viewport_ind + val, len(self.viewports))
        # highlight the new viewport
        if highlight: self.viewports[self.curr_viewport_ind].clear_color = [0.2, 0.2, 0.2, 1.0]
        print ('selected viewport: ', self.viewports[self.curr_viewport_ind].name)

    def ref_inc_ind(self, val=1, highlight=True):
        # only do anything if we have a current indicator viewport
        if self.viewports[self.curr_viewport_ind].name.startswith('ind'):
            if highlight: self.viewports[self.curr_viewport_ind].ref_coords[self.curr_indicator_ind*3]
            self.curr_indicator_ind = mod(self.curr_indicator_ind + val, self.viewports[self.curr_viewport_ind].num_refs)

    def add_viewport(self, batch, name='vp-0', coords=[0,0,150,150], scale_factors=[1.,1.,1],
                     for_ax=[2,-1], up_ax=[1,1], projection='perspective',
                     frustum=[-.1,.1,-.1,.1,.1,2.0], ref_pt_size=5,
                     ref_coords=[16.0,80.0,0.0,82.0,82.0,0.0,14.0,16.0,0.0,83.0,16.0,0.0],
                     pan=0.0, tilt=0.0, dutch=0.0):
        '''Add a new viewport class to the window list of viewports to draw'''
        # must have a unique name
        while any([name==vp.name for vp in self.viewports]):
            if name[-1].isdecimal():
                bname = name[:-1]
                num = name[-1] + 1
            else:
                bname = name
                num = 0
            name = '{}{}'.format(bname, num)
        # add the new viewport to the list
        self.viewports.append(Viewport_class(batch, name, coords,
                                             scale_factors, for_ax, up_ax,
                                             projection=projection,
                                             frustum=frustum,
                                             ref_pt_size=ref_pt_size,
                                             ref_coords=ref_coords,
                                             pan=pan, tilt=tilt, dutch=dutch))
        self.curr_viewport_ind = len(self.viewports) - 1
    
    def viewport_set_val(self, field, val, increment='set', viewport_ind=None):
        '''Set a value for all the viewports (but not refs)'''
        if viewport_ind is None or viewport_ind.startswith('curr'): # set to the current vp
            viewports = [self.viewports[self.curr_viewport_ind]]
        elif isinstance(viewport_ind, int):                         # single integer set to a vp
            viewport = [self.viewports[viewport_ind]]
        elif viewport_ind=='all':                    # set the value for all of them
            viewports = self.viewports
        elif viewport_ind=='all-ref':                # set the value for all of them but ref windows
            viewports = [vp for vp in self.viewports if not vp.projection.startswith('ref')]
        elif viewport_ind=='ref':                    # set the value for only the ref windows
            viewports = [vp for vp in self.viewports if vp.projection.startswith('ref')]
        if increment.startswith('inc'):
            increment = True
        else:
            increment = False
        for viewport in viewports:
            viewport.set_val(field, val, increment)

    def set_near(self, near):
        print ('near')
        self.viewport_set_val('near', near, 'set', 'all-ref')
        
    def set_far(self, far):
        self.viewport_set_val('far',  far,  'set', 'all-ref')

    def set_bg(self, color=[0.0, 0.0, 0.0, 1.0]):
        self.viewport_set_val('bg', color, 'set', 'all-ref')

    def toggle_fullscreen(self):
        '''Switch the fullscreen state on the current screen'''
        self.set_fullscreen(not self.fullscreen)

    def toggle_screen(self):
        '''Switch the active current display screen'''
        self.screen_number = (self.screen_number + 1)%len(self.screens)
        self.set_fullscreen(self.fullscreen, self.screens[self.screen_number])

    def set_viewport_projection(self, viewport_ind=0, projection=0, half=False):
        '''change the projection of this viewport:
        0 - orthographic
        1 - frustum (perspective)
        2 - orthographic for ref window, not the scene as a whole'''
        if projection == 0:
            self.viewports[viewport_ind].draw = self.viewports[viewport_ind].draw_ortho
            self.viewports[viewport_ind].sprite_pos = int((viewport_ind + 1)*1000)
            self.viewports[viewport_ind].orth_coords = self.viewports[viewport_ind].coords[:].copy()
            if half:
                self.viewports[viewport_ind].orth_coords[2]/=2
                self.viewports[viewport_ind].orth_coords[3]/=2
        elif projection == 1:
            self.viewports[viewport_ind].draw = self.viewports[viewport_ind].draw_perspective

    def set_ref(self, ref_ind, color, viewport_ind=-1):
        '''Set the color of a ref pt with a three tuple'''
        self.viewports[viewport_ind].ref_vl.colors[(ref_ind*3):(ref_ind*3 + 3)] = color

    def move_ref(self, ref_ind, ax_ind, viewport_ind=0):
        '''Set the pos of a ref pt with a three tuple'''
        # choose the proper viewport---usually there is only one
        ref_vp = [vp for vp in self.viewports if vp.name.startswith('ind')][viewport_ind]
        # swap the color of the color vertexes
        ref_vp.ref_colors[(ref_ind*3):(ref_ind*3 + 3)] = color

    # functions to set flash patterns for ref lights
    def set_flash_pattern(self, pattern, num_frames=0, ref_ind=0,
                          dist=10, col_1=255, col_2=96,
                          viewport_ind=0):
        '''Sets the flash pattern for a reference flasher. the pattern is a
        binary string, or the whole array if flash_pattern is not None.'''
        # choose the proper viewport---usually there is only one
        ref_vp = [vp for vp in self.viewports if vp.name.startswith('ind')][viewport_ind]
        if pattern is None:            #delete the entry
            ref_vp[ref_ind].pop(ref_ind)
        elif isinstance(pattern, ndarray):   #an array, just for playing
            ref_vp.flash_patterns[ref_ind] = pattern
        elif isinstance(pattern, str):       
            if pattern.startswith('0b'):      #a binary number, just for playing
                flash_pattern = zeros(num_frames, dtype='ubyte')
                for i,p in enumerate(pattern):
                    flash_pattern[(i+1)*dist] = col_1 if pattern[i] in '1b' else col_2
                    ref_vp.flash_patterns[ref_ind] = array(flash_pattern)
        
    
    def calc_flash_pattern(self, param, num_frames, ref_ind=0,
                           dist=10, col_1=255, col_2=96,
                           viewport_ind=0):
        '''Calculate the flash pattern needed to signal current states of the window'''
        if param == 'az':
            num = mod(arctan2(self.rot[2,2], self.rot[2,0])*180/pi - 90, 360)
            self.set_flash_pattern(bin(int(num)), num_frames, ref_ind, dist, col_1, col_2, viewport_ind)
        if param == 'el':
            num = 360-mod(arctan2(self.rot[2,1], sqrt(self.rot[2,2]**2 + self.rot[2,0]**2))*180/pi, 360)
            self.set_flash_pattern(bin(int(num)), num_frames, ref_ind, dist, col_1, col_2, viewport_ind)
        if param == 'start-stop':
            pat = zeros(num_frames, dtype='ubyte')
            pat[[1,-2]] == col_1
            self.set_flash_pattern(pat, num_frames, ref_ind, dist, col_1, col_2, viewport_ind)
            

    def flash_patterns(self, frame_ind, viewport_ind=0):
        '''Set all the flash patterns in the reference viewport to the current frame'''
        # choose the proper viewport---usually there is only one
        ref_vp = [vp for vp in self.viewports if vp.name.startswith('ind')][viewport_ind]
        for ref_ind in ref_vp.flash_patterns:
            color = ref_vp.flash_patterns[ref_ind][frame_ind]
            ref_vp.ref_vl.colors[(ref_ind*3):(ref_ind*3 + 3)] = (0, color, 0)

    ## alter position and heading of the viewpoint ##
    # alter position and heading directly
    def set_pos(self, pos):
        '''Set the x, y, z of the viewpoint'''
        self.pos = pos

    def set_rot(self, rot):
        '''Set the rotation matrix around the viewpoint'''
        self.rot = rot

    def set_rot_angles(self, yaw=0, pitch=0, roll=0):
        '''Set the rotation matrix around the viewpoint given specific angles.'''
        # make the 3 rotation matrices
        sint, cost = sin(yaw), cos(yaw)
        yaw_mat = array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
        sint, cost = sin(pitch), cos(pitch)
        pitch_mat = array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])
        sint, cost = sin(roll), cos(roll)
        roll_mat = array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
        # multiply them together
        # self.rot = dot(dot(yaw_mat, pitch_mat), roll_mat)
        self.rot = dot(dot(pitch_mat, roll_mat), yaw_mat)

    # alter position and heading relative to global axes
    def set_px(self, dis):
        '''Set the x position (left and right) of the viewpoint'''
        self.pos[0] = dis

    def set_py(self, dis):
        '''Set the y position (up and down) of the viewpoint'''
        self.pos[1] = dis

    def set_pz(self, dis):
        '''Set the z position (forward and backward) of the viewpoint'''
        self.pos[2] = dis

    def inc_px(self, dis):
        '''Increment the x position (left and right) of the viewpoint'''
        self.pos[0] += dis

    def inc_py(self, dis):
        '''Increment the y position (up and down) of the viewpoint'''
        self.pos[1] += dis

    def inc_pz(self, dis):
        '''Increment the z position (forward and backward) of the viewpoint'''
        self.pos[2] += dis

    def inc_rx(self, ang=0):
        '''Increment current heading around the global x axis'''
        sint, cost = sin(ang), cos(ang)
        mat = array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])
        dot(mat, self.rot, out=self.rot)
        
    def inc_ry(self, ang=0):
        '''Increment current heading around the global y axis'''
        sint, cost = sin(ang), cos(ang)
        mat = array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
        dot(mat, self.rot, out=self.rot)
        
    def set_ry(self, ang=0):
        '''Increment current heading around the global y axis'''
        sint, cost = sin(ang), cos(ang)
        mat = array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
        self.rot = mat
        # dot(mat, self.rot, out=self.rot)
        
    def inc_rz(self, ang=0):
        '''Increment current heading around the global z axis'''
        sint, cost = sin(ang), cos(ang)
        mat = array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
        dot(mat, self.rot, out=self.rot)

    # alter position and heading relative to current position and heading
    def inc_slip(self, dis):
        '''Move the viewpoint left and right, relative to heading'''
        self.pos += dis*dot(self.rot, array([1.,0.,0]))
        
    def inc_lift(self, dis):
        '''Move the viewpoint up and down, relative to heading'''
        self.pos += dis*dot(self.rot, array([0.,1.,0.]))
        
    def inc_thrust(self, dis):
        '''Move the viewpoint forward and backward, relative to heading'''
        # self.pos += dis*dot(self.rot, array([0.,0.,1.]))
        self.pos += dis*array([0.,0.,1.])
        
    def inc_pos(self, pos_delta):
        '''Increment the current position by a 3D vector'''
        self.pos += pos_delta

    def inc_pitch(self, ang=0):
        '''Increment current heading in pitch'''
        sint, cost = sin(ang), cos(ang)
        mat = array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])
        dot(self.rot, mat, out=self.rot)

    def inc_yaw(self, ang=0):
        '''Increment current heading in yaw'''
        sint, cost = sin(ang), cos(ang)
        mat = array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
        dot(self.rot, mat, out=self.rot)

    def set_yaw(self, ang=0):
        '''Increment current heading in yaw'''
        if not np.isnan(ang):
            self.yaw = ang
            sint, cost = sin(ang), cos(ang)
            mat = array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
            dot(self.rot, mat, out=self.rot)
        else:
            pass

    def inc_roll(self, ang=0):
        '''Increment current heading in roll'''
        sint, cost = sin(ang), cos(ang)
        mat = array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
        dot(self.rot, mat, out=self.rot)

    def set_roll(self, ang=0):
        '''Increment current heading in yaw'''
        if not np.isnan(ang):
            self.roll = ang
            sint, cost = sin(ang), cos(ang)
            rot = array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
            mat = array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
            dot(self.rot, mat, out=self.rot)
        else:
            pass

    # reset to center position and straight heading
    def reset_pos(self):
        '''Set pos back to 0,0,0'''
        self.pos = zeros((3))

    def reset_rot(self):
        '''Set heading back to identity matrix'''
        self.rot = identity(3)

    def reset_pos_rot(self):
        '''Set pos back to 0,0,0, heading back to identity matrix'''
        self.pos = zeros((3))
        self.rot = identity(3)

    def reset_frame(self, frame=0):
        self.frame = frame

    def save_cam(self, pos=True, rot=True, bg=True, near=True, far=True):
        '''Save the curret camera position and rotation, bg color, near and far settings.'''
        if pos: self.pos_saved = self.pos.copy()
        if rot: self.rot_saved = self.rot.copy()
        self.bg_saved, self.near_saved, self.far_saved = [], [], []
        for viewport in self.viewports:
            if bg: self.bg_saved.append(viewport.clear_color.copy())
            if near: self.near_saved.append(viewport.frustum[4])
            if far: self.far_saved.append(viewport.frustum[5])

    def restore_cam(self, pos=True, rot=True, bg=True, near=True, far=True):
        '''Save the curret camera position and rotation, bg color, near and far settings.'''
        if pos: self.pos = self.pos_saved.copy()
        if rot: self.rot = self.rot_saved.copy()
        for viewport in self.viewports:
            if bg: viewport.set_val('bg', self.bg_saved.pop(0))
            if near: viewport.set_val('near', self.near_saved.pop(0))
            if far: viewport.set_val('far', self.far_saved.pop(0))
        
    # query position and heading
    def get_pos(self):
        '''Get the x, y, z of the viewpoint'''
        return self.pos

    def get_yaw(self):
        '''Return the current window yaw value.'''
        return self.yaw

    def get_rot(self):
        '''Get the rotation matrix around the viewpoint'''
        return self.rot

    def record_start(self, params=['yaw', 'realtime']):
        self.record_params = params
        self.record = {}
        self.record_ind = 0
        for param in self.record_params:
            self.record[param] = []

    def record_frame(self):
        for param in self.record_params:
            if param == 'realtime':
                val = time.time()
            else:
                val = self.__getattribute__(param)
            self.record[param] += [val]
        self.record_ind += 1

    def record_stop(self, fn=None):
        """Store the recorded parameters as a structured numpy array."""
        dtypes = [(param, np.float64) for param in self.record_params]
        # arr = np.zeros((len(dtypes)+1, self.record_ind), dtype=dtypes)
        arr = np.zeros((len(dtypes)+1, self.record_ind))
        for num, (param) in enumerate(self.record_params):
            arr[num] = np.array(self.record[param])
        if len(self.record_params) == 1:
            arr = arr[0]
        if fn is None:
            return arr
        else:
            np.save(fn, arr)

    def save_png(self, prefix='frame'):
        # set the viewport to grab the whole window
        # (trying to grab only one viewport leads to strange behavior---
        #  only the first one is ever imaged, regardless of how which coords are specified)
        glViewport(0,0,self.w, self.h)
        self.bufm.get_color_buffer().save(f'{prefix}_{self.frame:06d}.png')
        
    def get_frame(self):
        test = glViewport(0,0,self.w, self.h)
        frame = self.bufm.get_color_buffer()
        return frame

    def str_to_key(self, s):
        '''Change a string to an appropriate key tuple,
        one key, zero or more modifiers seperated by spaces, any order, upper or lower case 
        mods: shift ctrl alt capslock numlock windows command option scrolllock function accel
        keys: A-Z, 0-9, BACKSPACE, TAB, LINEFEED, CLEAR, RETURN, ENTER, PAUSE, SCROLLLOCK
        SYSREQ, ESCAPE, SPACE, HOME, LEFT, UP, RIGHT, DOWN, PAGEUP, PAGEDOWN, END, BEGIN
        DELETE, SELECT, PRINT, EXECUTE, INSERT, UNDO, REDO, MENU, FIND, CANCEL, HELP
        BREAK, NUM_SPACE, NUM_TAB, NUM_ENTER, NUM_F1, NUM_F2, NUM_F3, NUM_F4, NUM_HOME
        NUM_LEFT, NUM_UP, NUM_RIGHT, NUM_DOWN, NUM_PRIOR, NUM_PAGE_UP, NUM_NEXT
        NUM_PAGE_DOWN, NUM_END, NUM_BEGIN, NUM_INSERT, NUM_DELETE, NUM_EQUAL
        NUM_MULTIPLY, NUM_ADD, NUM_SEPARATOR, NUM_SUBTRACT, NUM_DECIMAL, NUM_DIVIDE
        NUM_0, NUM_1, NUM_2, NUM_3, NUM_4, NUM_5, NUM_6, NUM_7, NUM_8, NUM_9, F1, F2, F3
        F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20
        EXCLAMATION, DOUBLEQUOTE, HASH, POUND, DOLLAR, PERCENT, AMPERSAND, APOSTROPHE
        PARENLEFT, PARENRIGHT, ASTERISK, PLUS, COMMA, MINUS, PERIOD, SLASH, COLON
        SEMICOLON, LESS, EQUAL, GREATER, QUESTION, AT, BRACKETLEFT, BACKSLASH
        BRACKETRIGHT, ASCIICIRCUM, UNDERSCORE, GRAVE, QUOTELEFT, BRACELEFT, BAR
        BRACERIGHT, ASCIITILDE
        '''
        k, mods = 0, 0
        for ss in s.split(' '):
            ssu = ss.upper()                  # constants in key are uppercase
            if 'MOD_' + ssu in key.__dict__:  # turn ctrl indo MOD_CTRL
                mods += key.__dict__['MOD_' + ssu] # add to mods
            elif len(ssu)==1 and ssu.isdigit(): # turn 5 into _5
                ssu = '_' + ssu
                k = key.__dict__['_' + ssu]
            elif ssu in key.__dict__:          
                k = key.__dict__[ssu]
        return (k, mods)
    
    def dict_key(self, key):
        '''Change a keypress with modifiers to an appropriate tuple for a dictionary key'''
        if isinstance(key, str):           # is this a string
            key = self.str_to_key(key)     # parse it for key and mods
        elif not hasattr(key, '__iter__'): # did we provide a (key modifier) tuple?
            key = (key, 0)                 # if not, make a tuple with 0 modifier
        elif len(key)>2:                   # are there multiple modifiers?
            key = (key[0], sum(key[1:]))   # if so, add them together
        return tuple(key)                  # in case we passed a list, change it to an immutable tuple

    def add_keypress_action(self, key, action, *args, **kwargs):
        '''Add a function to be executed once when a key is pressed'''
        self.add_key_action(1, key, action, *args, **kwargs)

    def add_keyhold_action(self, key, action, *args, **kwargs):
        '''Add a function to be executed continuously while a key is pressed'''
        self.add_key_action(inf, key, action, *args, **kwargs)

    def add_key_action(self, num_frames, key, action, *args, **kwargs):
        '''Add a function to be executed once when a key is pressed'''
        self.key_actions[self.dict_key(key)] = [num_frames, action, args, kwargs]
    
    def save_keys(self):
        '''Save the current key bindings, to later reset them to this state.
        This should let you assign keys without having to explicitly
        unbind them later
        '''
        self.key_actions_saved = self.key_actions.copy()

    def restore_keys(self):
        '''Restore key bindings to the last saved state'''
        self.key_actions = self.keyhold_actions_saved

    def remove_key_action(self, key):
        '''Free up a key press combination'''
        key = self.dict_key(key)
        if key in self.key_actions:
            del(self.key_actions[key])
        
    def remove_key_actions(self, keys):
        '''Free up a set of key press combination'''
        for key in keys:
            self.remove_key_action(key)
        
    def print_keypress_actions(self):
        items = sorted(self.key_actions.items())
        for keypress, action in items:
            keysymbol = key.symbol_string(keypress[0]).lstrip(' _')
            modifiers = key.modifiers_string(keypress[1]).replace('MOD_', '').replace('|', ' ').lstrip(' ')
            func, args, kwargs = action[0].__name__, action[1], action[2]
            print('{:<10} {:<6} --- {:<30}({}, {})'.format(modifiers, keysymbol, func, args, kwargs))

    def print_info(self):
        '''print information about everything'''
        print( 'pos:\n{}'.format(self.pos))
        print('rot\n{}'.format(self.rot))
        print('fps = {}\n'.format(pyglet.clock.get_fps()))
        for vp in self.viewports:
            print('viewport - {}'.format(vp.name))

    ##############
    ### ON KEY ###
    ##############
    def on_key_press(self, symbol, modifiers):
        '''Execute functions for a key press event'''
        # close the window (for when it has no visible close box)
        if symbol == key.PAUSE or symbol == key.BREAK:
            print('quitting now...')
            self.close()
            
        elif (symbol, modifiers) in self.key_actions:
            self.frame_actions.append(self.key_actions[(symbol, modifiers)])

        # if there were no hits, report which keys were pressed
        else:
            if symbol not in [65507, 65508, 65513, 65514, 65505, 65506]: #if it's not a common modifier pressed on its own
                print('No action for {} {} ({} {})'.format(key.modifiers_string(modifiers), key.symbol_string(symbol), modifiers, symbol)) #print whatever it was

    def on_key_release(self, symbol, modifiers):
        '''When a key is released, remove its action from the frame_actions list, if it is there'''
        if (symbol, modifiers) in self.key_actions and self.key_actions[symbol, modifiers][0]==inf:
            self.frame_actions.remove(self.key_actions[(symbol, modifiers)])

    ###############
    ### ON DRAW ###
    ###############
    def on_draw(self):
        '''Each frame, clear the whole area, draw each viewport, and execute any held key commands'''
        # first clear the whole screen with black
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glScissor(0,0,self.w,self.h)
        
        # then set the bg_color to clear each viewport
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.bg_color)

        # now let each viewport draw itself
        for viewport in self.viewports:
            viewport.draw(self.pos, self.rot)

        # execute any keypress frame action commands
        for num_frames, fun, args, kwargs in self.frame_actions:
            fun(*args, **kwargs)

        # filter the frame_action list to eliminate finished actions
        self.frame += 1
        self.frame_actions = [[num_frames - 1, action, args, kwargs] for num_frames, action, args, kwargs in self.frame_actions if num_frames > 1]

    def on_close(self):
        self.alive = 0


if __name__=='__main__':
    project = 0
    bg = [1., 1., 1., 1.]
    near, far = .01, 1.

    import holocube.hc5 as hc5


    hc5.window.start(project=project, bg_color=bg, near=near, far=far)
    
    hc5.window.add_key_action(key._0, self.abort, True)

    # run pyglet
    pyglet.app.run()
