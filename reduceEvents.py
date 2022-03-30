"""
Comparaison of functions to reduce temporally and spacially the events

Author: Amelie Gruel
Date: 08/2021 - 11/2021
"""

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
        sim_time=-1,   
        input_ev=np.zeros((0,4)),
        coord_t=3,
        div=2,
        width=-1,
        height=-1
    ):
        self.events = np.zeros((0,4))
        self.coord_t = coord_t
        self.coord_p = self.getPolarityIndex()
        
        self.input = input_ev.copy()
        # self.min_t = min(self.input[:, self.coord_t])
        # self.input[:, self.coord_t] = self.input[:, self.coord_t] - self.min_t

        self.div = div

        if width != -1 or height != -1:
            self.width_fullscale = width
            self.height_fullscale = height
        else: 
            self.width_fullscale, self.height_fullscale = self.getSensorSize()

        self.width_downscale, self.height_downscale = self.getDownscaledSensorSize()
        self.neg_pol = self.getNegativeEventsValue()
        
        if sim_time != -1:
            self.t = 0
            self.sim_time = sim_time
        else:
            self.sim_time = self.getTimeLength()


    # global functions

    def getSensorSize(self):
        return int(np.max(self.input[::,0]))+1,int(np.max(self.input[::,1]))+1

    def getTimeLength(self):
        return int(np.max(self.input[:,self.coord_t]))

    def getDownscaledSensorSize(self):
        return ceil(self.width_fullscale/self.div), ceil(self.height_fullscale/self.div)     # to keep modified ? 

    def newEvent(self,x,y,p,t):
        if self.coord_t == 2:
            return [x,y,t,p]
        elif self.coord_t == 3:
            return [x,y,p,t]
    
    def getPolarityIndex(self):
        return (set([2,3]) - set([self.coord_t])).pop()

    def getNegativeEventsValue(self):
        return np.unique(self.input[self.input[:,self.coord_p] < 1][:,self.coord_p]).item()

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
        self.reduce()

    
    def reduce(self):
        pass



### SPATIAL FUNELLING ###
# Reduction by funelling events spatially 

class SpatialFunnelling(Reduction):

    def __init__(self, sim_time=-1, input_ev=np.zeros((0, 4)), coord_t=3, div=2, width=-1, height=-1):
        """
        Arguments: 
        - events (numpy array): events to be reduced
        - coord_t (int): index of the timestamp coordinate 
        Optional arguments:
        - div (int): by how much to divide the events (spacial reduction)
        """
        super().__init__(sim_time, input_ev, coord_t, div, width, height)

    def reduce(self):
        self.input[:,0] = np.floor(self.input[:,0]/self.div)
        self.input[:,1] = np.floor(self.input[:,1]/self.div)
        _, idx=np.unique(self.input, axis=0, return_index=True)    
        
        self.updateEvents( self.input[np.sort(idx)] )





### EVENT COUNT ###

class EventCount(Reduction):

    def __init__(self, sim_time=-1, input_ev=np.zeros((0, 4)), coord_t=3, div=2, width=-1, height=-1, threshold=1, plot=False):
        """
        Arguments: 
        - events (numpy array): events to be reduced
        - coord_t (int): index of the timestamp coordinate 
        Optional arguments:
        - div (int): by how much to divide the events (spacial reduction)
        - threshold (float): ????
        """
        super().__init__(sim_time, input_ev, coord_t, div, width, height)
        self.threshold = threshold
        self.plot = plot

        self.input = self.input[self.input[:,self.coord_t].argsort()]  # this method needs the events to be sorted according to the timestamps 
        
        # parameters specific to the event count method
        self.fullscale_state = np.zeros((self.width_fullscale, self.height_fullscale))
        self.downscale_state = np.zeros((self.width_downscale, self.height_downscale))
        self.lastDownscaleEvents = np.array( [ [None]*self.height_downscale ]*self.width_downscale )
        if self.plot:
            self.EvCount = [[[[0,0]]]*self.height_downscale]*self.width_downscale
        self.action = {
            1            :  1,
            self.neg_pol : -1
        }


    def reduce(self):
        for ev in self.input :    # parcourir en fonction du temps ? => pas forcément plus court 
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

    def __init__(self, sim_time=-1, input_ev=np.zeros((0, 4)), coord_t=3, div=2, width=-1, height=-1,
        threshold=1, cubic_interpolation=True,  # if False, the log luminance is interpolated as a polyline
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
        super().__init__(sim_time, input_ev, coord_t, div, width, height)

        self.sim_time += 1
        self.threshold = threshold
        self.cubic_interpolation = cubic_interpolation
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
        for n_pixel in range(self.width_downscale*self.height_downscale):
            downscale_w, downscale_h = np.unravel_index(n_pixel, (self.width_downscale, self.height_downscale))
            
            T_ = np.linspace(0,self.sim_time, self.sensitivity)
            L_ = np.zeros((self.sensitivity, 0))

            # get points coordonates from fullscale pixels correspond to n_pixel
            w_min, w_max, h_min, h_max = getFullscaleCoord(downscale_w, downscale_h, self.div)
            for fullscale_w in range(w_min, w_max):
                for fullscale_h in range(h_min, h_max):
                    ev_pixel = self.input[ np.logical_and( self.input[:,0]==fullscale_w , self.input[:,1]==fullscale_h ) ]
                    t, levels = ev2points(ev_pixel, 1, self.coord_p)
                    t, levels = extrapolateLevels(t, levels, self.sim_time)
                    
                    if self.cubic_interpolation:
                        cubic_inter = self.s.PchipInterpolator(t, levels)
                        L = np.array(cubic_inter(T_))
                    else :
                        L = np.interp(T_, t, levels) 
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
        """
        if len(timestamps) == 0:
            timestamps = [0]
            levels = [0]
        return timestamps, levels
        """
        try : 
            last_lvl = levels[-1]
        except IndexError:
            last_lvl = 0
        
        if len(timestamps) < 4:
            data = [(0,-2,0), (1,-1,0), (2,0,0), (len(timestamps)+3, self.sim_time, last_lvl)]
        else : 
            data = [(0,0,0), (len(timestamps)+1, self.sim_time, last_lvl)]

        for idx, value_t, value_l in data:
            timestamps = np.insert(timestamps, idx, value_t)
            levels = np.insert(levels, idx, value_l)
        _,order = np.unique(timestamps, return_index=True)
        timestamps = timestamps[order]
        levels = levels[order]
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
    


"""
### SPIKING NEURAL NETWORKS ###
# Reduction by downscaling events using a spiking neural network

def ev2spikes(events,coord_t, width, height):
    print("\nTranslating events to spikes... ")
    if not 1<coord_t<4:
        raise ValueError("coord_t must equals 2 or 3")
    
    coord_t-=2
    
    spikes=[[] for _ in range(width*height)]
    for x,y,*r in tqdm(events):
        spikes[int(x)*height+int(y)].append(float(r[coord_t]))
    print("Translation done\n")
    return spikes


def spikes2ev(spikes, width, height, coord_t, polarity=1):
    events = np.zeros((0,4))
    for n in range(len(spikes)):
        x,y = np.unravel_index(n, (width, height))
        pixel_events = np.array([ newEvent(x,y,polarity,t.item(), coord_t) for t in spikes[n]]) 
        try : 
            events = np.vstack((
                events,
                pixel_events
            ))
        except ValueError:
            pass

    events = events[events[:,coord_t].argsort()]
    return events


def runSim(sim, input_spikes, sim_length, div, coord_t, neg_pol, width_fullscale, height_fullscale, keep_polarity, density, mutual=True, plot=True):
    sim.setup(timestep=0.01)
    
    width_downscale, height_downscale = getDownscaledSensorSize(width_fullscale, height_fullscale, div)
    if keep_polarity :
        fullscale_size = width_fullscale * height_fullscale * 2
        downscale_size = width_downscale * height_downscale * 2
    else : 
        fullscale_size = width_fullscale * height_fullscale
        downscale_size = width_downscale * height_downscale
    
    print("Network initialisation...")
    fullscale_events = sim.Population(
        fullscale_size,
        sim.SpikeSourceArray(spike_times=input_spikes),
        label="Full scale events"
    )
    
    if keep_polarity:
        n=2
    else : 
        n=1
    i=0
    c=0
    subregions_fullscale_events = []
    while i < n:
        for X in range(width_downscale):
            for Y in range(height_downscale):

                subregion_coordonates = np.array([
                    np.ravel_multi_index( (x,y) , (width_fullscale, height_fullscale) ) + c
                    for x in range(div*X, div*(X+1)) if x < width_fullscale
                    for y in range(div*Y, div*(Y+1)) if y < height_fullscale
                ])
                
                subregions_fullscale_events.append(
                    sim.PopulationView(fullscale_events, subregion_coordonates)
                )
        
        c=int(fullscale_size/2)
        i+=1

    downscale_events = sim.Population(
        downscale_size,
        sim.IF_cond_exp(),
        label="Down scale events"
    )
    downscale_events.record(("spikes","v"))
    print("Populations done")

    fullscale2downscale = []
    mutual_inhibition = []
    c=int(downscale_events.size/2)
    for n in range(downscale_events.size):
        fullscale2downscale.append( sim.Projection(
            subregions_fullscale_events[n],
            sim.PopulationView(downscale_events, [n]),
            connector=sim.AllToAllConnector(),
            synapse_type=sim.StaticSynapse(weight=density),
            receptor_type="excitatory",
            label="Excitatory connection between fullscale and downscale events"
        ))
    
        if mutual:
            if n < c: 
                neuron = sim.PopulationView(downscale_events, [n+c])
            else : 
                neuron = sim.PopulationView(downscale_events, [n-c])
            mutual_inhibition.append( sim.Projection(
                subregions_fullscale_events[n],
                neuron,
                connector=sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=density),
                receptor_type="inhibitory",
                label="Inhibitory connection between fullscale and downscale events"
            ))

    
    print("Connection done\n")

    class visualiseTime(object):
        def __init__(self, sampling_interval):
            self.interval = sampling_interval

        def __call__(self, t):
            print(t)
            return t + self.interval
    visualise_time = visualiseTime(sampling_interval=100.0)

    print("Start downscaling...")
    sim.run(sim_length, callbacks=[visualise_time])
    print("Downscaling done")

    spikes = downscale_events.get_data("spikes").segments[0].spiketrains
    
    if keep_polarity :
        pos_events = spikes2ev(spikes[:int(len(spikes)/2)], width_downscale, height_downscale, coord_t, polarity=1)
        neg_events = spikes2ev(spikes[int(len(spikes)/2):], width_downscale, height_downscale, coord_t, polarity=neg_pol)
    
    if plot:
        v = downscale_events.get_data("v").segments[0].filter(name='v')[0]
        if keep_polarity:
            v_pos = list(e.item() for e in v[:,0].reshape(-1))
            v_neg = list(e.item() for e in v[:,1].reshape(-1))

    sim.end()

    if keep_polarity:
        events = np.vstack((pos_events, neg_events))
        events = events[events[:,coord_t].argsort()]
    else :
        events = spikes2ev(spikes, width_downscale, height_downscale, coord_t)
    
    if plot:
        return events, v_pos, v_neg
    return events


def SNN_downscale(
    events,
    coord_t,
    div=4,
    density=0.2, #????
    keep_polarity=True,
    mutual=True,
    simulator_capacity=5000,
    time_reduce=True,
    plot=True
):
    
    Arguments: 
    - events (numpy array): events to be reduced
    - coord_t (int): index of the timestamp coordinate 
    Optionnal arguments:
    - div (int): by how much to divide the events (spacial reduction)
    - density (float between 0 and 1): density of the downscaling
    - keep_polarity (boolean): wether to keep the polarity of events or ignore them (all downscaled events are positive)
    
    import pyNN.nest as sim

    downscaled_events = np.zeros((0,4))
    if time_reduce:
        events[:,coord_t] *= 0.001
    coord_p = getPolarityIndex(coord_t)
    neg_pol = getNegativeEventsValue(events, coord_p)
    width_fullscale,height_fullscale=getSensorSize(events)

    if plot:
        v_pos = []
        v_neg = []
    
    last_time = 0
    nb_sim = int( np.max(events[:,coord_t]) // simulator_capacity + 1)
    print("Downscaling with Spiking Neural Network Pooling will run "+str(nb_sim)+" simulations")
    for s in range(nb_sim):
        print("\n> Starting simulation "+str(s+1)+"...")
        spikes = events[ np.logical_and(
            events[:,coord_t] > s*simulator_capacity,
            events[:,coord_t] <= (s+1)*simulator_capacity, 
        ) ]
        spikes[:,coord_t] -= last_time
        sim_length=getTimeLength(spikes, coord_t)
        print("Length simulation: "+str(sim_length)+" ts")

        if keep_polarity:
            pos_events = ev2spikes(spikes[spikes[:,coord_p] > 0], coord_t, width_fullscale, height_fullscale)
            neg_events = ev2spikes(spikes[spikes[:,coord_p] < 1], coord_t, width_fullscale, height_fullscale)
            spikes = pos_events+neg_events
        else : 
            spikes = ev2spikes(spikes, coord_t, width_fullscale, height_fullscale)
        
        downscaled_spikes = runSim(sim, spikes, sim_length, div, coord_t, neg_pol, width_fullscale, height_fullscale, keep_polarity, density, mutual)
        if plot :
            downscaled_spikes, vp, vn = downscaled_spikes
            v_pos = v_pos + vp
            v_neg = v_neg + vn

        downscaled_spikes[:,coord_t] += last_time
        downscaled_events = np.vstack((downscaled_events, downscaled_spikes))

        last_time = sim_length

    if time_reduce:
        events[:,coord_t] *= 1000
        downscaled_events[:,coord_t] *= 1000
    
    if plot:
        return downscaled_events, v_pos, v_neg
    return downscaled_events
"""


### GLOBAL FUNCTIONS ###

def getSensorSize(events):
    return int(np.max(events[::,0]))+1,int(np.max(events[::,1]))+1

def getTimeLength(events, coord_t):
    return int(np.max(events[:,coord_t]))

def getPolarityIndex(coord_t):
    return (set([2,3]) - set([coord_t])).pop()

def getDownscaledSensorSize(width, height, div):
    return ceil(width/div), ceil(height/div)     # to keep modified ? 

def newEvent(x,y,p,t, coord_t):
    if coord_t == 2:
        return [x,y,t,p]
    elif coord_t == 3:
        return [x,y,p,t]





### FUNELLING ###
# Reduction by funelling events spatially and/or temporally

def reduce(
    events,
    coord_t,
    div=2,
    spacial=True,
    temporal=True
):
    """
    Arguments: 
    - events (numpy array): events to be reduced
    - coord_t (int): index of the timestamp coordinate 
    Optionnal arguments:
    - div (int): by how much to divide the events (spacial reduction)
    - spacial (bool): reduce spacially the events, by dividing the width and height of the captor by the div
    - temporal (bool): reduce temporally the events, by rounding thz timestamps to the ms
    """
    events = events.copy()
    if spacial:
        events[:,0] = np.floor(events[:,0]/div)
        events[:,1] = np.floor(events[:,1]/div)
    if temporal:
        events[:,coord_t] = np.floor(events[:,2])
    _, idx=np.unique(events, axis=0, return_index=True)    
    return events[np.sort(idx)]




### SPIKING NEURAL NETWORKS ###
# Reduction by downscaling events using a spiking neural network

def ev2spikes(events,coord_t, width, height):
    print("\nTranslating events to spikes... ")
    if not 1<coord_t<4:
        raise ValueError("coord_t must equals 2 or 3")
    
    coord_t-=2
    
    spikes=[[] for _ in range(width*height)]
    for x,y,*r in tqdm(events):
        spikes[int(x)*height+int(y)].append(float(r[coord_t]))
    print("Translation done\n")
    return spikes


def spikes2ev(spikes, width, height, coord_t, polarity=1):
    events = np.zeros((0,4))
    for n in range(len(spikes)):
        x,y = np.unravel_index(n, (width, height))
        pixel_events = np.array([ newEvent(x,y,polarity,t.item(), coord_t) for t in spikes[n]]) 
        try : 
            events = np.vstack((
                events,
                pixel_events
            ))
        except ValueError:
            pass

    events = events[events[:,coord_t].argsort()]
    return events


def runSim(sim, input_spikes, sim_length, div, coord_t, neg_pol, width_fullscale, height_fullscale, keep_polarity, density, mutual=True, plot=True):
    sim.setup(timestep=0.01)
    
    width_downscale, height_downscale = getDownscaledSensorSize(width_fullscale, height_fullscale, div)
    if keep_polarity :
        fullscale_size = width_fullscale * height_fullscale * 2
        downscale_size = width_downscale * height_downscale * 2
    else : 
        fullscale_size = width_fullscale * height_fullscale
        downscale_size = width_downscale * height_downscale
    
    print("Network initialisation...")
    fullscale_events = sim.Population(
        fullscale_size,
        sim.SpikeSourceArray(spike_times=input_spikes),
        label="Full scale events"
    )
    
    if keep_polarity:
        n=2
    else : 
        n=1
    i=0
    c=0
    subregions_fullscale_events = []
    while i < n:
        for X in range(width_downscale):
            for Y in range(height_downscale):

                subregion_coordonates = np.array([
                    np.ravel_multi_index( (x,y) , (width_fullscale, height_fullscale) ) + c
                    for x in range(div*X, div*(X+1)) if x < width_fullscale
                    for y in range(div*Y, div*(Y+1)) if y < height_fullscale
                ])
                
                subregions_fullscale_events.append(
                    sim.PopulationView(fullscale_events, subregion_coordonates)
                )
        
        c=int(fullscale_size/2)
        i+=1

    downscale_events = sim.Population(
        downscale_size,
        sim.IF_cond_exp(),
        label="Down scale events"
    )
    downscale_events.record(("spikes","v"))
    print("Populations done")

    fullscale2downscale = []
    mutual_inhibition = []
    c=int(downscale_events.size/2)
    for n in range(downscale_events.size):
        fullscale2downscale.append( sim.Projection(
            subregions_fullscale_events[n],
            sim.PopulationView(downscale_events, [n]),
            connector=sim.AllToAllConnector(),
            synapse_type=sim.StaticSynapse(weight=density),
            receptor_type="excitatory",
            label="Excitatory connection between fullscale and downscale events"
        ))
    
        if mutual:
            if n < c: 
                neuron = sim.PopulationView(downscale_events, [n+c])
            else : 
                neuron = sim.PopulationView(downscale_events, [n-c])
            mutual_inhibition.append( sim.Projection(
                subregions_fullscale_events[n],
                neuron,
                connector=sim.AllToAllConnector(),
                synapse_type=sim.StaticSynapse(weight=density),
                receptor_type="inhibitory",
                label="Inhibitory connection between fullscale and downscale events"
            ))

    
    print("Connection done\n")

    class visualiseTime(object):
        def __init__(self, sampling_interval):
            self.interval = sampling_interval

        def __call__(self, t):
            print(t)
            return t + self.interval
    visualise_time = visualiseTime(sampling_interval=100.0)

    print("Start downscaling...")
    sim.run(sim_length, callbacks=[visualise_time])
    print("Downscaling done")

    spikes = downscale_events.get_data("spikes").segments[0].spiketrains
    
    if keep_polarity :
        pos_events = spikes2ev(spikes[:int(len(spikes)/2)], width_downscale, height_downscale, coord_t, polarity=1)
        neg_events = spikes2ev(spikes[int(len(spikes)/2):], width_downscale, height_downscale, coord_t, polarity=neg_pol)
    
    if plot:
        v = downscale_events.get_data("v").segments[0].filter(name='v')[0]
        if keep_polarity:
            v_pos = list(e.item() for e in v[:,0].reshape(-1))
            v_neg = list(e.item() for e in v[:,1].reshape(-1))

    sim.end()

    if keep_polarity:
        events = np.vstack((pos_events, neg_events))
        events = events[events[:,coord_t].argsort()]
    else :
        events = spikes2ev(spikes, width_downscale, height_downscale, coord_t)
    
    if plot:
        return events, v_pos, v_neg
    return events


def SNN_downscale(
    events,
    coord_t,
    div=4,
    density=0.2, #????
    keep_polarity=True,
    mutual=True,
    simulator_capacity=5000,
    time_reduce=True,
    plot=True
):
    """
    Arguments: 
    - events (numpy array): events to be reduced
    - coord_t (int): index of the timestamp coordinate 
    Optionnal arguments:
    - div (int): by how much to divide the events (spacial reduction)
    - density (float between 0 and 1): density of the downscaling
    - keep_polarity (boolean): wether to keep the polarity of events or ignore them (all downscaled events are positive)
    """

    import pyNN.nest as sim

    downscaled_events = np.zeros((0,4))
    if time_reduce:
        events[:,coord_t] *= 0.001
    coord_p = getPolarityIndex(coord_t)
    neg_pol = getNegativeEventsValue(events, coord_p)
    width_fullscale,height_fullscale=getSensorSize(events)

    if plot:
        v_pos = []
        v_neg = []
    
    last_time = 0
    nb_sim = int( np.max(events[:,coord_t]) // simulator_capacity + 1)
    print("Downscaling with Spiking Neural Network Pooling will run "+str(nb_sim)+" simulations")
    for s in range(nb_sim):
        print("\n> Starting simulation "+str(s+1)+"...")
        spikes = events[ np.logical_and(
            events[:,coord_t] > s*simulator_capacity,
            events[:,coord_t] <= (s+1)*simulator_capacity, 
        ) ]
        spikes[:,coord_t] -= last_time
        sim_length=getTimeLength(spikes, coord_t)
        print("Length simulation: "+str(sim_length)+" ts")

        if keep_polarity:
            pos_events = ev2spikes(spikes[spikes[:,coord_p] > 0], coord_t, width_fullscale, height_fullscale)
            neg_events = ev2spikes(spikes[spikes[:,coord_p] < 1], coord_t, width_fullscale, height_fullscale)
            spikes = pos_events+neg_events
        else : 
            spikes = ev2spikes(spikes, coord_t, width_fullscale, height_fullscale)
        
        downscaled_spikes = runSim(sim, spikes, sim_length, div, coord_t, neg_pol, width_fullscale, height_fullscale, keep_polarity, density, mutual)
        if plot :
            downscaled_spikes, vp, vn = downscaled_spikes
            v_pos = v_pos + vp
            v_neg = v_neg + vn

        downscaled_spikes[:,coord_t] += last_time
        downscaled_events = np.vstack((downscaled_events, downscaled_spikes))

        last_time = sim_length

    if time_reduce:
        events[:,coord_t] *= 1000
        downscaled_events[:,coord_t] *= 1000
    
    if plot:
        return downscaled_events, v_pos, v_neg
    return downscaled_events




### METHODE GUILLAUME ###
# Reduction by blabla (à définir)+= action[ev[coord_p]]

def getDownscaleCoord(x,y,div):
    return floor(x/div), floor(y/div)

def getFullscaleCoord(x,y,div):
    return x*div, (x+1)*div, y*div, (y+1)*div

def event_count(
    events,
    coord_t,
    div=4,
    threshold=1,
    plot=False
):
    """
    Arguments: 
    - events (numpy array): events to be reduced
    - coord_t (int): index of the timestamp coordinate 
    Optionnal arguments:
    - div (int): by how much to divide the events (spacial reduction)
    - threshold (float): ????
    """
    
    events = events[events[:,coord_t].argsort()]  # this method needs the events to be sorted according to the timestamps 

    # width_fullscale, height_fullscale = getSensorSize(events)
    width_fullscale = height_fullscale = 128
    width_downscale, height_downscale = getDownscaledSensorSize(width_fullscale, height_fullscale, div)
    coord_p = getPolarityIndex(coord_t)
    # neg_pol = getNegativeEventsValue(events, coord_p)
    neg_pol = 0
    
    fullscale_state = np.zeros((width_fullscale, height_fullscale))
    downscale_state = np.zeros((width_downscale, height_downscale))
    lastDownscaleEvents = np.array( [ [None]*height_downscale ]*width_downscale )
    downscaled_events = np.zeros((0,4))

    if plot:
        EvCount = [[[[0,0]]]*height_downscale]*width_downscale

    action = {
        1       :  1,
        neg_pol : -1
    }
    for ev in events :    # parcourir en fonction du temps ? => pas forcément plus court 
        fullscale_state[ int(ev[0]), int(ev[1]) ] += action[ev[coord_p]]
        
        # old and new levels in downscale state
        downscale_w,downscale_h = getDownscaleCoord(ev[0], ev[1], div)
        old_level = downscale_state[downscale_w,downscale_h]
        w_min, w_max, h_min, h_max = getFullscaleCoord(downscale_w, downscale_h, div)
        new_level = np.sum( fullscale_state[ w_min:w_max , h_min:h_max ] ) / (div**2)

        event = False
        if not new_level % threshold and new_level > old_level :
            if lastDownscaleEvents[downscale_w, downscale_h] != "neg":
                polarity = 1
                event = True
            lastDownscaleEvents[downscale_w, downscale_h] = "pos"
        elif not new_level % threshold and new_level < old_level:
            if lastDownscaleEvents[downscale_w, downscale_h] != "pos":
                polarity = neg_pol
                event = True
            lastDownscaleEvents[downscale_w, downscale_h] = "neg"
        
        if event:
            downscaled_events = np.vstack((
                downscaled_events,
                newEvent(downscale_w, downscale_h, polarity, ev[coord_t], coord_t)
            ))
        
        downscale_state[downscale_w, downscale_h] = new_level
        if plot:
            if ev[coord_t] == EvCount[downscale_w][downscale_h][-1][0] :
                EvCount[downscale_w][downscale_h][-1][-1] = new_level
            else:
                EvCount[downscale_w][downscale_h].append([ev[coord_t], new_level])

    if plot : 
        return downscaled_events, EvCount
    return downscaled_events




### LOG LUMINANCE ###
# Reduction by computing the mean of log luminance curve of each pixel in the region to be downscaled, 
# then outputing the events as intersection of this mean curve with the treshold levels

def ev2points(events, threshold, coord_p):
    ordinate = 0     # we arbitrarly set the starting level's ordinate at 0: each event's ordinate will then be considered as x*threshold + start 
    action = {
        1 :  threshold,
        0 : -threshold,
        -1: -threshold
    }
    points = []
    for ev in events:
        ordinate += action[ev[coord_p]]
        points.append([ev[3], ordinate])
    if not len(points):
        return np.zeros((2,0))
    else :
        return np.array(points).T


def extrapolateLevels(timestamps, levels, timelength):
    try : 
        last_lvl = levels[-1]
    except IndexError:
        last_lvl = 0
    
    if len(timestamps) < 4:
        data = [(0,-2,0), (1,-1,0), (2,0,0), (len(timestamps)+3, timelength, last_lvl)]
    else : 
        data = [(0,0,0), (len(timestamps)+1, timelength, last_lvl)]

    for idx, value_t, value_l in data:
        timestamps = np.insert(timestamps, idx, value_t)
        levels = np.insert(levels, idx, value_l)
    _,order = np.unique(timestamps, return_index=True)
    timestamps = timestamps[order]
    levels = levels[order]
    return timestamps, levels


def extractIntersections(timestamps, y_curve, threshold):
    # compute where y changes direction
    y_directions = np.sign(np.diff(y_curve))
    y_extrema = 1 + np.where(np.diff(y_directions) != 0)[0]

    # get segments aka groups of [t,y] where y is monotonic
    t_segments = np.split(timestamps, y_extrema)
    y_segments = np.split(y_curve, y_extrema)

    
    y_levels = np.arange(floor(np.min(y_curve)), ceil(np.max(y_curve))+1, threshold)
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

def logluminance_downscale(
    events,
    coord_t,
    div=4,
    threshold=1,
    cubic_interpolation=True,  # if False, the log luminance is interpolated as a polyline
    sensitivity=500, # name ?????
    plot=False,
    nb_subplots=8
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

    from scipy.interpolate import PchipInterpolator

    downscaled_events = np.zeros((0,4))
    width_fullscale, height_fullscale = getSensorSize(events)
    width_downscale, height_downscale = getDownscaledSensorSize(width_fullscale, height_fullscale, div)
    timelength = getTimeLength(events, coord_t)+1
    coord_p = getPolarityIndex(coord_t)
    neg_pol = getNegativeEventsValue(events, coord_p)

    if plot:
        import matplotlib.pyplot as plt
        if nb_subplots > 1:
            _, axs = plt.subplots(int(nb_subplots/2),int(nb_subplots/2))
            w_ = int(width_downscale/2 - nb_subplots/4)
            h_ = int(height_downscale/2 - nb_subplots/4)
        else : 
            _, fig = plt.subplots()

    for n_pixel in range(width_downscale*height_downscale):
        downscale_w, downscale_h = np.unravel_index(n_pixel, (width_downscale, height_downscale))
        
        T_ = np.linspace(0,timelength, sensitivity)
        L_ = np.zeros((sensitivity, 0))

        if plot and nb_subplots > 1:
            try :
                fig = axs[downscale_w - w_, downscale_h - h_]
            except IndexError:
                pass

        # get points coordonates from fullscale pixels correspond to n_pixel
        w_min, w_max, h_min, h_max = getFullscaleCoord(downscale_w, downscale_h, div)
        for fullscale_w in range(w_min, w_max):
            for fullscale_h in range(h_min, h_max):
                ev_pixel = events[ np.logical_and( events[:,0]==fullscale_w , events[:,1]==fullscale_h ) ]
                t, levels = ev2points(ev_pixel, 1, coord_p)
                t, levels = extrapolateLevels(t, levels, timelength)
                
                if cubic_interpolation:
                    cubic_inter = PchipInterpolator(t, levels)
                    L = np.array(cubic_inter(T_))
                else :
                    L = np.interp(T_, t, levels) 
                L_ = np.concatenate(( L_, L.reshape(-1,1)), axis=1)
                
                if plot and ((nb_subplots > 1 and downscale_w in range(w_, w_ + int(nb_subplots/2)) and downscale_h in range(h_, h_ + int(nb_subplots/2))) or (nb_subplots <=1 and downscale_w == int(nb_subplots/2) and downscale_h == int(nb_subplots/2))):
                    fig.set_xlim([0,timelength])
                    fig.plot(T_, L)
                    # fig.hlines( range(int(np.max(L_)) + 2), 0, timelength, linestyles="dotted", color="gray" )

        mean_level = np.mean(L_, axis=1)
        if plot and ((nb_subplots > 1 and downscale_w in range(w_, w_ + int(nb_subplots/2)) and downscale_h in range(h_, h_ + int(nb_subplots/2))) or (nb_subplots <=1 and downscale_w == int(nb_subplots/2) and downscale_h == int(nb_subplots/2))):
            fig.plot(T_, mean_level, "--", color="black")

        # get events from levels
        T_events, L_events = extractIntersections(T_, mean_level, threshold)
        old_state = 0
        last_event = None

        for t, new_state in zip(T_events, L_events):
            
            event = False
            if not new_state % threshold and new_state > old_state :
                if last_event != "neg":
                    polarity = 1
                    event = True
                last_event = "pos"
            elif not new_state % threshold and new_state < old_state:
                if last_event != "pos":
                    polarity = neg_pol
                    event = True
                last_event = "neg"
            elif new_state == old_state and old_state != None:
                last_event = (set(["pos", "neg"]) - set([last_event])).pop()
            if event:
                downscaled_events = np.vstack((
                    downscaled_events,
                    newEvent(downscale_w, downscale_h, polarity, t, coord_t) 
                ))
            
            old_state = new_state

    if plot:
        plt.show()
    return downscaled_events