"""
Comparison of functions to reduce temporally and spatially the events

Author: Amelie Gruel
Date: 11/2021 - 02/2022
"""
import numpy as np
from math import e, floor, ceil
import time


### GLOBAL CLASS WITH GENERAL FUNCTIONS ###

class Reduction():

    def __init__(
        self,
        sim_time=1e+6,    # 1 second
        input_ev=np.zeros((0,4)),
        neg_pol=0,
        coord_t=3,
        div=2,
        width=128,
        height=128
    ):
        self.events = np.zeros((0,4))
        self.coord_t = coord_t
        self.coord_p = self.getPolarityIndex()
        
        self.input = input_ev.copy()
        min_t = min(self.input[:, self.coord_t])
        self.input[:, self.coord_t] = self.input[:, self.coord_t] - min_t

        self.input_t = np.zeros((0,4))
        self.div = div
        self.width_fullscale = width
        self.height_fullscale = height
        self.width_downscale, self.height_downscale = self.getDownscaledSensorSize()
        self.neg_pol = neg_pol
        
        self.t = 0
        self.sim_time = sim_time

    # global functions

    def getTimeLength(self, events):
        return int(np.max(events[:,self.coord_t]))

    def getDownscaledSensorSize(self):
        return ceil(self.width_fullscale/self.div), ceil(self.height_fullscale/self.div)     # to keep modified ? 

    def newEvent(self,x,y,p,t):
        if self.coord_t == 2:
            return [x,y,t,p]
        elif self.coord_t == 3:
            return [x,y,p,t]
    
    def getPolarityIndex(self):
        return (set([2,3]) - set([self.coord_t])).pop()

    def getFullscaleCoord(self,x,y):
        return x*self.div, min((x+1)*self.div, self.width_fullscale), y*self.div, min((y+1)*self.div, self.height_fullscale)

    def getDownscaleCoord(self,x,y):
        return floor(x/self.div), floor(y/self.div)

    def computationTime(self):
        start=time.time()
        
        self.reduce()

        stop=time.time()
        print(stop - start)

    # run simulations
    def updateEvents(self, input):
        self.events = np.vstack((
            self.events,
            input
        ))

    def updateT(self):
        self.t += 1
    
    def runSimulation(self):
        while self.t <= self.sim_time:
            print(self.t)
            # if self.t % 1000 == 0:
            #     print(self.t)
            self.input_t = self.input[self.input[:,self.coord_t] == self.t]
            self.reduce()
            # self.computationTime()
            self.updateT()
    
    def reduce(self):
        pass



### SPATIAL FUNELLING ###
# Reduction by funelling events spatially 

class SpatialFunnelling(Reduction):

    def __init__(self, sim_time=1e+6, input_ev=np.zeros((0, 4)), neg_pol=0, coord_t=3, div=2, width=128, height=128):
        """
        Arguments: 
        - events (numpy array): events to be reduced
        - coord_t (int): index of the timestamp coordinate 
        Optional arguments:
        - div (int): by how much to divide the events (spacial reduction)
        """
        super().__init__(sim_time, input_ev, neg_pol, coord_t, div, width, height)

    def reduce(self):
        self.input_t[:,0] = np.floor(self.input_t[:,0]/self.div)
        self.input_t[:,1] = np.floor(self.input_t[:,1]/self.div)
        _, idx=np.unique(self.input_t, axis=0, return_index=True)    
        
        self.updateEvents( self.input_t[np.sort(idx)] )





### EVENT COUNT ###

class EventCount(Reduction):

    
    def __init__(self, sim_time=1e+6, input_ev=np.zeros((0, 4)), neg_pol=0, coord_t=3, div=2, width=128, height=128,threshold=1, plot=False):
        """
        Arguments: 
        - events (numpy array): events to be reduced
        - coord_t (int): index of the timestamp coordinate 
        Optional arguments:
        - div (int): by how much to divide the events (spacial reduction)
        - threshold (float): ????
        """
        super().__init__(sim_time, input_ev, neg_pol, coord_t, div, width, height)
        self.threshold = threshold
        self.plot = plot

        self.input = self.input[self.input[:,self.coord_t].argsort()]  # this method needs the events to be sorted according to the timestamps 
        
        # parameters specific to the event count method
        self.fullscale_state = np.zeros((self.width_fullscale, self.height_fullscale))
        self.downscale_state = np.zeros((self.width_downscale, self.height_downscale))
        self.lastDownscaleEvents = np.array( [ [None]*self.height_downscale ]*self.width_downscale )
        # self.downscaled_events = np.zeros((0,4))
        if self.plot:
            self.EvCount = [[[[0,0]]]*self.height_downscale]*self.width_downscale
        self.action = {
            1            :  1,
            self.neg_pol : -1
        }


    def reduce(self):
        for ev in self.input_t :    # parcourir en fonction du temps ? => pas forcément plus court 
            self.fullscale_state[ int(ev[0]), int(ev[1]) ] += self.action[ev[self.coord_p]]
            
            # old and new levels in downscale state
            downscale_w,downscale_h = self.getDownscaleCoord(ev[0], ev[1])
            old_level = self.downscale_state[downscale_w,downscale_h]
            w_min, w_max, h_min, h_max = self.getFullscaleCoord(downscale_w, downscale_h)
            new_level = np.sum( self.fullscale_state[ w_min:w_max , h_min:h_max ] ) / (self.div**2)

            event = False
            if not new_level % self.threshold and new_level > old_level :
                if self.lastDownscaleEvents[downscale_w, downscale_h] != "neg":
                    polarity = 1
                    event = True
                self.lastDownscaleEvents[downscale_w, downscale_h] = "pos"
            elif not new_level % self.threshold and new_level < old_level:
                if self.lastDownscaleEvents[downscale_w, downscale_h] != "pos":
                    polarity = self.neg_pol
                    event = True
                self.lastDownscaleEvents[downscale_w, downscale_h] = "neg"
            
            if event:
                self.updateEvents( self.newEvent(downscale_w, downscale_h, polarity, ev[self.coord_t]) )
            
            self.downscale_state[downscale_w, downscale_h] = new_level
            if self.plot:
                if ev[self.coord_t] == self.EvCount[downscale_w][downscale_h][-1][0] :
                    self.EvCount[downscale_w][downscale_h][-1][-1] = new_level
                else:
                    self.EvCount[downscale_w][downscale_h].append([ev[self.coord_t], new_level])



### LOG LUMINANCE ###
# Reduction by computing the mean of log luminance curve of each pixel in the region to be downscaled, 
# then outputing the events as intersection of this mean curve with the treshold levels

class LogLuminance(Reduction):

    def __init__(self, sim_time=1e+6, input_ev=np.zeros((0, 4)), neg_pol=0, coord_t=3, div=2, width=128, height=128, 
        threshold=1, cubic_interpolation=True,  # if False, the log luminance is interpolated as a polyline
        algo_constant=True,
        sensitivity=500
    ):
        """
        Arguments: 
        - events (numpy array): events to be reduced
        - coord_t (int): index of the timestamp coordinate 
        Optionnal arguments:
        - div (int): by how much to divide the events (spacial reduction)
        - cubic_interpolation (bool): wether to interpolate the log luminance as a polynomial curve ('True') or as a polyline ('False')
        - threshold (float): ???
        """
        super().__init__(sim_time, input_ev, neg_pol, coord_t, div, width, height)
        self.threshold = threshold
        self.cubic_interpolation = cubic_interpolation
        self.algo_constant = algo_constant
        self.sensitivity = sensitivity
        
        self.action = {
            1 :  self.threshold,
            0 : -self.threshold,
            -1: -self.threshold
        }

        # memory of previous points calculated at full scale
        self.points = [[[[-2,-1],[0,0]] for _ in range(self.width_fullscale)] for _ in range(self.height_fullscale)]
        
        import scipy.interpolate as s
        self.s = s


    def reduce(self):

        for ev in self.input_t:
            fullscale_w = ev[0]
            fullscale_h = ev[1]
            t, levels = self.ev2points(ev)
            t, levels = self.extrapolateLevels(t, levels)
            if t[0] not in self.points[fullscale_w][fullscale_h][0]:
                self.points[fullscale_w][fullscale_h][0].append( t[0] )
                self.points[fullscale_w][fullscale_h][1].append( levels[0] )
                
            if self.algo_constant :
                downscale_w, downscale_h = self.getDownscaleCoord(fullscale_w, fullscale_h)
                self.get_llcurve(downscale_w, downscale_h)

        if not self.algo_constant:
            for n_pixel in range(self.width_downscale*self.height_downscale):
                downscale_w, downscale_h = np.unravel_index(n_pixel, (self.width_downscale, self.height_downscale))
                self.get_llcurve(downscale_w, downscale_h)
    
    def get_llcurve(self, downscale_w, downscale_h):
            
        T_ = np.linspace(self.t-3,self.t, self.sensitivity)
        L_ = np.zeros((self.sensitivity, 0))

        # get points coordinates from fullscale pixels correspond to n_pixel
        w_min, w_max, h_min, h_max = self.getFullscaleCoord(downscale_w, downscale_h)
        for fullscale_w in range(w_min, w_max):
            for fullscale_h in range(h_min, h_max):
                    
                if self.cubic_interpolation:
                    cubic_inter = self.s.PchipInterpolator(self.points[fullscale_w][fullscale_h][0][-3:], self.points[fullscale_w][fullscale_h][1][-3:])
                    L = np.array(cubic_inter(T_))
                else :
                    L = np.interp(T_, self.points[fullscale_w][fullscale_h][0][-3:], self.points[fullscale_w][fullscale_h][1][-3:]) 
                L_ = np.concatenate(( L_, L.reshape(-1,1)), axis=1)

        mean_level = np.mean(L_, axis=1)

        # get events from levels
        T_events, L_events = self.extractIntersections(T_, mean_level)
        old_state = 0
        last_event = None

        for t, new_state in zip(T_events, L_events):
            
            event = False
            if not new_state % self.threshold and new_state > old_state :
                if last_event != "neg":
                    polarity = 1
                    event = True
                last_event = "pos"
            elif not new_state % self.threshold and new_state < old_state:
                if last_event != "pos":
                    polarity = self.neg_pol
                    event = True
                last_event = "neg"
            elif new_state == old_state and old_state != None:
                last_event = (set(["pos", "neg"]) - set([last_event])).pop()
            
            if event:
                self.updateEvents(self.newEvent(downscale_w, downscale_h, polarity, t))
            
            old_state = new_state


    def ev2points(self, ev):
        ordinate = 0     # we arbitrarily set the starting level's ordinate at 0: each event's ordinate will then be considered as x*threshold + start 
        points = []
        ordinate += self.action[ev[self.coord_p]]
        points.append([ev[3], ordinate])
        if not len(points):
            return np.zeros((2,0))
        else :
            return np.array(points).T


    def extrapolateLevels(self, timestamps, levels):
        if len(timestamps) == 0:
            timestamps = [0]
            levels = [0]
        return timestamps, levels


    def extractIntersections(self, timestamps, y_curve):
        # compute where y changes direction
        y_directions = np.sign(np.diff(y_curve))
        y_extrema = 1 + np.where(np.diff(y_directions) != 0)[0]

        # get segments aka groups of [t,y] where y is monotonic
        t_segments = np.split(timestamps, y_extrema)
        y_segments = np.split(y_curve, y_extrema)

        
        y_levels = np.arange(floor(np.min(y_curve)), ceil(np.max(y_curve))+1, self.threshold)
        T_events = []
        Y_events = []
        for y_ev in y_levels:
            t_ev = [np.interp(y_ev, y_seg, t_seg) if y_seg[0] < y_seg[-1] else np.interp(y_ev, y_seg[::-1], t_seg[::-1]) for y_seg, t_seg in zip(y_segments, t_segments) if min(y_seg) <= y_ev <= max(y_seg)]
            T_events += t_ev
            Y_events += [y_ev]*len(t_ev) 
        
        # sort according to timestamps
        T_events, order = np.unique(T_events, return_index=True)
        Y_events = np.array(Y_events)[order]

        return T_events, Y_events



class StochasticStructural(Reduction):

    def __init__(self, sim_time=1e+6, input_ev=np.zeros((0, 4)), neg_pol=0, coord_t=3, div=50, width=128, height=128):
        super().__init__(sim_time, input_ev, neg_pol, coord_t, div, width, height)

        self.nb_events_selected = int(div * len(self.input) / 100)
        self.idx = np.random.choice(len(self.input), size=self.nb_events_selected, replace=False)
        self.current_idx = 0

    def reduce(self):
        for ev in self.input_t:
            if self.current_idx in self.idx:
                self.updateEvents(ev)
            self.current_idx += 1

class DeterministicStructural(Reduction):

    def __init__(self, sim_time=1e+6, input_ev=np.zeros((0, 4)), neg_pol=0, coord_t=3, div=2, width=128, height=128):
        super().__init__(sim_time, input_ev, neg_pol, coord_t, div, width, height)
        self.current_idx = 0

    def reduce(self):
        for ev in self.input_t:
            if not self.current_idx % self.div:
                self.updateEvents(ev)
            self.current_idx += 1


### TEMPORAL FUNELLING ###
# Reduction by funelling events temporally 

class TemporalFunnelling(Reduction):

    def __init__(self, sim_time=1e+6, input_ev=np.zeros((0, 4)), neg_pol=0, coord_t=3, div=2, width=128, height=128):
        """
        Arguments: 
        - events (numpy array): events to be reduced
        - coord_t (int): index of the timestamp coordinate 
        Optional arguments:
        - div (int): by how much to divide the events (spacial reduction)
        """
        super().__init__(sim_time, input_ev, neg_pol, coord_t, div, width, height)


    def reduce(self):
        for ev in self.input_t:
            ev[self.coord_t] = np.floor(ev[self.coord_t] * self.div)
            if ev not in self.events:
                self.updateEvents(ev)