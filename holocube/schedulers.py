# run a series of exps with pyg and ard

import pyglet
# pyglet.lib.load_library('abin')
pyglet.options['debug_lib'] = True 
from pyglet.gl import * #this will overwrite 'resize' if it comes first
from pyglet.window import key
import inspect
from os.path import isdir
import os
from numpy import *
from distutils.version import LooseVersion #to make sure numpy is at least 1.8
import sys
import importlib
import pdb

class Test():
    '''Hold all the commands for a test or rest.'''
    def __init__(self, name, num_frames, starts=[], middles=[], ends=[], exp=None, pos=0):
        self.name = name
        self.num_frames = num_frames
        self.starts = [[start[0], start[1:]] for start in starts]
        self.ends = [[end[0], end[1:]] for end in ends]
        self.mids = [[mid[0], mid[1:]] for mid in middles]
        self.exp = exp
        self.pos = pos
        if LooseVersion(__version__) >= LooseVersion('1.8'): #proper way to compare version strings. Or StrictVersion if we know it will be formed according to certain rules
            self.do_frame = self.do_frame_new
        else:
            self.do_frame = self.do_frame_old

    def __repr__(self):
        return '{} (Test class)'.format(self.name)

    def do_ends(self):
        '''Do the end arguments for terminating a test early'''
        for func, args in self.ends:
            func(*args)
        return False

    def do_frame_new(self, frame_num): #for newer versions of numpy indexing
        """Run every command in the starts, mids, or ends sections, depending
        on current frame_ind. For the middles, the argument can be:
        1. a function itself, that gets dynamically evaluated without
        arguments, 2. a numpy array, that gets indexed to the current
        frame, or 3. a plain argument, which is used for every frame.

        """
        if frame_num == 0:                   # first frame
            for func, arg_list in self.starts:
                try:
                    func(*arg_list)
                except:
                    breakpoint()
            return True
        elif frame_num < self.num_frames:    # each frame in the middle
            for func, arg_list in self.mids:
                func(*[arg() if hasattr(arg, '__call__') else
                       take(arg, frame_num, mode='wrap', axis=0) if isinstance(arg, ndarray) else
                       arg for arg in arg_list])
            return True
        else:
            for func, arg_list in self.ends:     # final frame
                func(*arg_list)
            return False

    def do_frame_old(self, frame_num): #for older versions of numpy indexing
        """Run every command in the starts, mids, or ends sections, depending
        on current frame_ind. For the middles, the argument can be:
        1. a function itself, that gets dynamically evaluated without
        arguments, 2. a numpy array, that gets indexed to the current
        frame, or 3. a plain argument, which is used for every frame.

        """
        if frame_num == 0:
            for func, args in self.starts:
                func(*args)
            return True
        elif frame_num < self.num_frames:
            for func, args in self.mids:
                func(*[arg() if hasattr(arg, '__call__') else
                       take(arg, [frame_num], mode='wrap', axis=0) if isinstance(arg, ndarray) else
                       arg for arg in args])
            return True
        else:
            for func, args in self.ends:
                func(*args)
            return False

    
class Experiment():
    '''Hold all the parameters of an experiment, which consists of multiple tests.'''
    def __init__(self, name, starts=[], ends=[]):
        self.name = name
        self.tests = [] #each test is a list of start, middle, and end funct and their parameters
        self.num_tests = 0
        self.rest = [] #rest is like a single test to be performed in between each test
        self.experiment = [] #this is just start and end funcs to execute before and after the whole experiment
        name = '%s'%('pre-' + self.name)
        self.starts = Test(name, 1, starts, [], [])
        name = '%s'%('post-' + self.name)
        self.ends = Test(name, 1, [], [], ends)

    def __repr__(self):
        return '%s (Experiment class)'%(self.name)

    def add_test(self, name=None, num_frames=0, starts=[], middles=[], ends=[]):
        '''Add a test (list of funcs and args), to the list of tests.'''
        if name==None: name = '{} - {:>2}'.format(self.name, len(self.tests) + 1)
        self.tests.append(Test(name, num_frames, starts, middles, ends, exp=self, pos=len(self.tests)))
        self.num_tests = len(self.tests)

    def add_rest(self, name=None, num_frames=0, starts=[], middles=[], ends=[]):
        '''Put the special test called rest into its own slot.'''
        if name==None:
            name = '%s - %d'%(self.name, len(self.tests) + 1)
        self.rest.append(Test(name, num_frames, starts, middles, ends))


class Scheduler():
    '''schedules a set of experiments with bar tracking in between.'''
    def __init__(self):
        self.exps = [] #holds the instances of class Experiment
        self.idles = []
        self.frame = 0       #the frame number in a rest or test
        self.test_list = []  #tests, rests, or idles to do

    def start(self, window, randomize=True, freq=120, default_rest_time=3, beep_ind=-1, beep_file=None):
        '''Start the scheduler (not an experiment)'''
        self.freq = freq

        self.window = window
        self.randomize = randomize #do we randomize experiment order
        self.default_rest_frames = default_rest_time*freq
        if beep_file:
            self.beep = pyglet.media.load(beep_file)
            self.beep_ind = beep_ind
        else: self.beep_ind = 0

        # all the keys
        self.window.add_keypress_action(key._1, self.begin_exp, 0)
        self.window.add_keypress_action(key._2, self.begin_exp, 1)
        self.window.add_keypress_action(key._3, self.begin_exp, 2)
        self.window.add_keypress_action(key._4, self.begin_exp, 3)
        self.window.add_keypress_action(key._5, self.begin_exp, 4)
        self.window.add_keypress_action(key._6, self.begin_exp, 5)
        self.window.add_keypress_action(key._7, self.begin_exp, 6)
        self.window.add_keypress_action(key._8, self.begin_exp, 7)
        self.window.add_keypress_action(key._9, self.begin_exp, 8)
        self.window.add_keypress_action(key._0, self.begin_exp, 9)
        self.window.add_keypress_action((key._1, key.MOD_CTRL), self.reload_exp, 0)
        self.window.add_keypress_action((key._2, key.MOD_CTRL), self.reload_exp, 1)
        self.window.add_keypress_action((key._3, key.MOD_CTRL), self.reload_exp, 2)
        self.window.add_keypress_action((key._4, key.MOD_CTRL), self.reload_exp, 3)
        self.window.add_keypress_action((key._5, key.MOD_CTRL), self.reload_exp, 4)
        self.window.add_keypress_action((key._6, key.MOD_CTRL), self.reload_exp, 5)
        self.window.add_keypress_action((key._7, key.MOD_CTRL), self.reload_exp, 6)
        self.window.add_keypress_action((key._8, key.MOD_CTRL), self.reload_exp, 7)
        self.window.add_keypress_action((key._9, key.MOD_CTRL), self.reload_exp, 8)
        self.window.add_keypress_action((key._0, key.MOD_CTRL), self.reload_exp, 9)
        self.window.add_keypress_action(key.R, self.toggle_randomize)
        self.window.add_keypress_action(key.BACKSPACE, self.abort_exp, True)
        self.window.add_keypress_action(key.HOME, self.print_keys, True)
        self.window.add_keypress_action(key.QUOTELEFT, self.change_idle, True)
        self.window.add_keypress_action(key.Q, self.close)

        # add a default idle that does nothing
        self.idles.append(Experiment('idle_do_nothing'))
        self.idles[-1].add_rest(None, inf, [], [], [])
        print ('')
        print ('\nKey assignments:')
        print ('{:<8} - {}'.format('Home', 'Print key assignments'))
        print ('{:<8} - {}'.format('BS', 'Abort experiment'))
        print ('{:<8} - {}'.format('Q', 'Quit'))
        print ('{:<8} - {} - current state = {}'.format('R', 'Toggle randomize', self.randomize))
        print ('')

        # now start the frames
        self.test_list = [self.idles[0].rest[0]]
        pyglet.clock.schedule_interval(self.show_frame, 1./self.freq)
        pyglet.options['vsync'] = True
        pyglet.clock.MIN_SLEEP = .005

    def load_dir(self, dir_name='experiments', suffix=('exp.py', 'rest.py')):
        '''load a file with experiments and rests'''
        if not isdir(dir_name): # if we don't see this directory, complain
            print ("Can\'t find directory named {}".format(dir_name))
            print(f"Currently inside {os.getcwd()}")
        else:
            # if not dir_name.endswith('/'): dir_name = dir_name + '/' # if we do see the directory, be sure it ends with a slash
            paths = [fn[:-3] for fn in os.listdir(dir_name) if fn.endswith(suffix) and not fn.startswith(('.', '#'))] # pick the files that are exps or rests
            paths.sort(key=lambda p: '0' if p.endswith('rest') else '1' + p) # sort the pathnames, rests first, then exps
            # home_dir = os.getcwd()
            # os.chdir(dir_name)
            # sys.path.append(dir_name)
            # for path in paths:
            #     try:
            #         importlib.import_module(path)
            #     except:
            #         breakpoint()
            # os.chdir(home_dir)
            sys.path.append(os.path.join(os.getcwd(), dir_name[:-1]))
            for path in paths:
                try:
                    importlib.import_module(f"{dir_name}.{path}")
                except:
                    print(f"Failed to import {path} from {dir_name}")
                    # give option to retry import for troubleshooting purposes
                    resp = None
                    while resp not in ['y', 'n']:
                        resp = input("Retry import? (y/n): ")
                    if resp == 'y':
                        importlib.import_module(f"{dir_name}.{path}")

    def add_exp(self, name=None, starts=[], ends=[]):
        if name==None:
            name = inspect.getouterframes(inspect.currentframe())[1][1] #get the file path that called save_exp
            name = name.rsplit('/')[-1].rsplit('.')[0]                  #get its name without '.py'
        self.exps.append(Experiment(name, starts, ends))
        print ('{:<8} - {}'.format(len(self.exps), name))
    
    def add_test(self, num_frames, starts, middles, ends):
        self.exps[-1].add_test(None, num_frames, starts, middles, ends)

    def add_rest(self, num_frames, starts, middles, ends):
        self.exps[-1].add_rest(None, num_frames, starts, middles, ends)

    def add_idle(self, num_frames, starts, middles, ends):
        name = inspect.getouterframes(inspect.currentframe())[1][1] #get the file path that called save_exp
        name = name.rsplit('/')[-1].rsplit('.')[0]                  #get its name without '.py'
        self.idles.append(Experiment(name))
        self.idles[-1].add_rest(None, num_frames, starts, middles, ends)
        print ('{:<8} - {}'.format('`', name))

    def reload_exp(self, number):
        oldexp = self.exps.pop(number - 1)
        name = oldexp.name
        print (name)
        from experiments import name
        self.exps.insert(ind, self.exps.pop(-1))

    def begin_exp(self, exp_ind):
        '''Start an experiment'''

        # stop any exp that's already running?
        self.abort_exp(printit=False)
        
        # is there an experiment loaded?
        if exp_ind >= len(self.exps):
            print ('\nOnly {} experiments loaded\n'.format(len(self.exps)))
            self.print_keys()
            return -1

        exp = self.exps[exp_ind]

        # determine the order
        if self.randomize: order = random.permutation(exp.num_tests)
        else: order = arange(exp.num_tests)

        # do we have a built in rest frame?
        if len(exp.rest) == 0:
            exp.add_rest(self.idles[0].rest[0].name,
                         self.default_rest_frames,
                         #ugly hack to get rid of extra nesting
                         [[s[0],s[1:][0][0]] for s in self.idles[0].rest[0].starts],
                         [[s[0],s[1:][0][0]] for s in self.idles[0].rest[0].mids],
                         [[s[0],s[1:][0][0]] for s in self.idles[0].rest[0].ends])
        rest=exp.rest[0]
    
        # build the list of tests
        # put the experiment starts in the list
        self.test_list = [exp.starts]
        for i, j in enumerate(order):
            # add the rest period
            self.test_list.append(rest)
            # label the upcoming test onscreen by making a test class that only does that
            self.test_list.append(Test('write', 1, [[self.print_test_data, (exp.tests[j].name, i+1, exp.num_tests)]]))
            if self.beep_ind and i == range(len(order))[self.beep_ind]:
                self.test_list.append(Test('beep', 1, [[self.play_beep, True]]))
            # add the exp test
            self.test_list.append(exp.tests[j])
        self.test_list.append(Test('done', 1, [[self.end_exp, exp_ind]], [], [],))
        # put the experiment ends in the list
        self.test_list.append(exp.ends)

        # start the experiment
        print ('')
        print(' begin {} {} {} '.format(exp_ind + 1, self.exps[exp_ind].name, order).center(80, '#'))
        self.frame = 0
        self.window.frame = 0

    def show_frame(self, dt=0):
        '''Execute everything to display a frame, called continuously.'''
        if self.test_list[0].do_frame(self.frame): #the frame is still inbounds
            self.frame += 1                        #so add one
        else:                                      #or it is exceeded
            self.pop_test_list()                   #so pop to the next test

    def end_exp(self, exp_ind):
        '''Finish an experiment'''
        print (' end {} {} '.format(exp_ind + 1, self.exps[exp_ind].name).center(80, '_'))
        print ('')
            
    def pop_test_list(self, clear=False):
        '''Pop the first test off the list, but add the current idle test if the list is ever empty.'''
        self.test_list.pop(0)
        self.frame = 0
        # if the list is empty, add on the current default recess Experiment
        if clear or not self.test_list:
            self.test_list = [self.idles[0].rest[0]]
                
    def abort_exp(self, printit=True):
        if printit: print (' Abort '.center(80, 'X'))
        self.test_list[0].do_ends() #end the running test (without reaching the final frame)
        if len(self.test_list)>1:
            self.test_list[-1].do_ends()#this should end the running exp, without reaching the final test
        self.pop_test_list(clear=True)

    def change_idle(self, num=1):
        '''Pop the first rest and put it at the end.'''
        self.idles.append(self.idles.pop(0))
        print ('{:<8} - {}'.format('new rest', self.idles[0].name))
        self.abort_exp(printit=False)

    def toggle_randomize(self):
        '''toggle the state of self.randomize'''
        if self.randomize: self.randomize = 0
        else: self.randomize = 1
        print("Randomize state: {}".format(self.randomize))

    def get_frame(self):
        '''Report the current frame, for another function to use'''
        return self.frame

    def print_keys(self, printit=True):
        '''Print the key assignments for each experiment.'''
        print ('\nKey assignments:\n')
        print ('{:<8} - {}'.format('Home', 'Print key assignments'))
        print ('{:<8} - {}'.format('BS', 'Abort experiment'))
        print ('{:<8} - {} - current state = {}'.format('R', 'Toggle randomize', self.randomize))
        print ('')
        for i, idle in enumerate(self.idles):
            print ('{:<8} - {}'.format('`', idle.name))
        print ('')
        for i,exp in enumerate(self.exps):
            print ('{:<8} - {}'.format(i+1, exp.name))
        print ('')

    def print_test_data(self, info):
        print (' {} - {:>2}/{} '.format(*info))

    def print_test(self, string=''):
        '''Print anything.'''
        print (string)

    def play_beep(self, play=True):
        self.beep.play()
        self.beep._is_queued = False #until this bug is fixed

    def close(self):
        self.window.close()
                    
if __name__=='__main__':
    import holocube.hc5 as hc
    s = Scheduler()
    s.start(hc.window, freq=2)
