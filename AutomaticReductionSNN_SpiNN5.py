from posixpath import join
from os import walk, path, makedirs
import time
import numpy as np
from math import ceil
from tqdm import tqdm
from datetime import datetime as d

# 

class SpikeRecorder(object):
    
    def __init__(self, sampling_interval, pop):
        self.interval = sampling_interval
        self.population = pop
        
        self._spikes = [[] for _ in self.population.size]

    def __call__(self, t):
        if t > 0:
            lastSpikes = map(
                lambda x: x[-1].item() if len(x) > 0 else 0, 
                self.population.get_data("spikes").segments[0].spiketrains
            )
            i = 0
            for s in lastSpikes:
                self._spikes[i].append(s)
                i += 1
        print(self._spikes)
        return t+self.interval

# FUNCTIONS

def saveAsNPZ(events, repertory_path, file_name):
    if not path.exists(repertory_path):
        makedirs(repertory_path)
    np.savez(
        join( repertory_path, file_name ),
        x=events[:,0],
        y=events[:,1],
        p=events[:,2],
        t=events[:,3],
    )

def getSensorSize(events):
    return int(np.max(events[::,0]))+1,int(np.max(events[::,1]))+1


def getDownscaledSensorSize(width, height, div):
    return ceil(width/div), ceil(height/div)     # to keep modified ? 

def getTimeLength(events, coord_t):
    return int(np.max(events[:,coord_t]))

def getPolarityIndex(coord_t):
    return (set([2,3]) - set([coord_t])).pop()

def getNegativeEventsValue(events, coord_p):
    return np.unique(events[events[:,coord_p] < 1][:,coord_p]).item()

def newEvent(x,y,p,t, coord_t):
    if coord_t == 2:
        return [x,y,t,p]
    elif coord_t == 3:
        return [x,y,p,t]

def ev2spikes(events,coord_t, width, height):
    # print("\nTranslating events to spikes... ")
    if not 1<coord_t<4:
        raise ValueError("coord_t must equals 2 or 3")
    
    coord_t-=2
    
    spikes=[[] for _ in range(width*height)]
    for x,y,*r in events: # tqdm(events):
        coord = int(np.ravel_multi_index( (int(x),int(y)) , (width, height) ))
        spikes[coord].append(float(r[coord_t]))
    # print("Translation done\n")
    return spikes


def spikes2ev(spikes, width, height, coord_t, polarity=1):
    events = np.zeros((0,4))
    for n in range(len(spikes)):
        x,y = np.unravel_index(n, (width, height))
        # if n < 10:
        #     print(n,x,y,"-", [t.item() for t in spikes[n]])
        pixel_events = np.array([ newEvent(x,y,polarity,t.item() - 1e3*i, coord_t) for t in spikes[n] if t.item() > 1e3*i]) 
        try : 
            events = np.vstack((
                events,
                pixel_events
            ))
        except ValueError:
            pass

    events = events[events[:,coord_t].argsort()]
    return events

def computationTime(input, fullscale_events, downscale_events, Div, method_time, time_factor):
    start=time.time()
    ev=SNN_downscale(
        input, 
        3,
        fullscale_events,
        downscale_events,
        div=Div,
        keep_polarity=True,
        time_reduce=True,
        time_factor=time_factor
    )
    stop=time.time()
    method_time += stop-start
    return ev, method_time


def init_SNN(sim, spikes, div, density=0.2, keep_polarity=True, mutual=True):

    width_fullscale,height_fullscale=getSensorSize(spikes)
    width_downscale, height_downscale = getDownscaledSensorSize(width_fullscale, height_fullscale, div)
    if keep_polarity :
        fullscale_size = width_fullscale * height_fullscale * 2
        downscale_size = width_downscale * height_downscale * 2
    else : 
        fullscale_size = width_fullscale * height_fullscale
        downscale_size = width_downscale * height_downscale
    # print("!!!!!!!! ", fullscale_size, downscale_size)

    # print("Network initialisation...")
    
    fullscale_events = sim.Population(
        fullscale_size,
        sim.SpikeSourceArray(spike_times=[]),
        label="Full scale events"
    )

    if keep_polarity:
        n=2
    else:
        n=1
    excitatory_connections = []
    inhibitory_connections = []
    for X in range(width_downscale):
        for Y in range(height_downscale*n):
            
            if Y < height_downscale:
                mutual_c = height_downscale
            else : 
                mutual_c = -height_downscale

            idx = np.ravel_multi_index( (X,Y) , (width_downscale, height_downscale*n) ) 
            excitatory_connections += [
                (
                    np.ravel_multi_index( (x,y) , (width_fullscale, height_fullscale*n) ) , 
                    idx
                )
                for x in range(int(div*X), int(div*(X+1))) if x < width_fullscale
                for y in range(int(div*Y), int(div*(Y+1))) if y < height_fullscale*n
            ]
            
            if mutual: 
                inhibitory_connections += [
                    (
                        np.ravel_multi_index( (x,y) , (width_fullscale, height_fullscale*n) ) , 
                        idx + mutual_c
                    )
                    for x in range(int(div*X), int(div*(X+1))) if x < width_fullscale
                    for y in range(int(div*Y), int(div*(Y+1))) if y < height_fullscale*n
                ]

    downscale_events = sim.Population(
        downscale_size,
        sim.IF_cond_exp(),
        label="Down scale events"
    )
    downscale_events.record(("spikes","v"))
    # print("Populations done")

    excitatory_fullscale2downscale = sim.Projection(
        fullscale_events,
        downscale_events,
        connector=sim.FromListConnector(excitatory_connections),
        synapse_type=sim.StaticSynapse(weight=density),
        receptor_type="excitatory",
        label="Excitatory connection between fullscale and downscale events"
    )

    if mutual:
        inhibitory_fullscale2downscale = sim.Projection(
            fullscale_events,
            downscale_events,
            connector=sim.FromListConnector(inhibitory_connections),
            synapse_type=sim.StaticSynapse(weight=density),
            receptor_type="inhibitory",
            label="Inhibitory connection between fullscale and downscale events"
        )
    
    # print("Connection done\n")
    # print("!!!!!!!! ", fullscale_size, downscale_size)

    return fullscale_events, downscale_events



def SNN_downscale(
    events,
    coord_t,
    fullscale_events,
    downscale_events,
    div=4,
    keep_polarity=True,
    time_reduce=True,
    time_factor=0.001
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

    spikes = events.copy()

    sim_length = 1e6
    if time_reduce:
        spikes[:,coord_t] *= time_factor
        sim_length *= time_factor
    spikes[:,coord_t] = spikes[:,coord_t] - min(spikes[:,coord_t])
    coord_p = getPolarityIndex(coord_t)
    neg_pol = getNegativeEventsValue(spikes, coord_p)
    width_fullscale,height_fullscale=getSensorSize(spikes)

    # print("\n> Starting simulation ...")
    # # sim_length=getTimeLength(spikes, coord_t)
    # print("Length simulation: "+str(sim_length)+" ts")
    # print("Size data",width_fullscale, height_fullscale, width_fullscale*height_fullscale)

    if keep_polarity:
        pos_events = ev2spikes(spikes[spikes[:,coord_p] > 0], coord_t, width_fullscale, height_fullscale)
        neg_events = ev2spikes(spikes[spikes[:,coord_p] < 1], coord_t, width_fullscale, height_fullscale)
        spikes = pos_events+neg_events
    else : 
        spikes = ev2spikes(spikes, coord_t, width_fullscale, height_fullscale)
    
    # print("input spikes")
    # for i in spikes[:10]:
    #     print(i)
    
    # width_downscale, height_downscale = getDownscaledSensorSize(width_fullscale, height_fullscale, div)
    width_downscale, height_downscale = (ceil(128/div), ceil(128/div))
    # print("Downsize data",width_downscale, height_downscale, width_downscale*height_downscale)
      
    ###########################################################################################################################################

    fullscale_events.set(spike_times=spikes)

    class visualiseTime(object):
        def __init__(self, sampling_interval):
            self.interval = sampling_interval

        def __call__(self, t):
            print(t, time.time())
            return t + self.interval
    # visualise_time = visualiseTime(sampling_interval=100.0)

    # outputSpikes = SpikeRecorder(sampling_interval=1.0, pop=downscale_events)
    sim.run(sim_length) #, callbacks=[outputSpikes])

    spikes = downscale_events.get_data("spikes").segments[-1].spiketrains
    # spikes = outputSpikes._spikes
    
    if keep_polarity :
        pos_events = spikes2ev(spikes[:int(len(spikes)/2)], width_downscale, height_downscale, coord_t, polarity=1)
        neg_events = spikes2ev(spikes[int(len(spikes)/2):], width_downscale, height_downscale, coord_t, polarity=neg_pol)
        events = np.vstack((pos_events, neg_events))
        events = events[events[:,coord_t].argsort()]
    else :
        events = spikes2ev(spikes, width_downscale, height_downscale, coord_t)
    
    ###########################################################################################################################################

    events = np.vstack((np.zeros((0,4)), events))  # handles case where no spikes produced by simulation

    if time_reduce:
        events[:,coord_t] *= 1000

    return events


# main
import pyNN.spiNNaker as sim
sim.setup(timestep=1)

dividers = [
    2,
    4,
    # 5,
    # 8,
    10,
    25,
    50
]

densities = [
    # 0.05,
    # 0.1,
    # 0.15,
    0.2 #=> already done
    # 0.25,
    # 0.3,
    # 0.35,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7
]

# par defaut chaque ts correspond a 1e6 second, cad a 1 microsecond
time_reduction_factors = [
    # 1e-6,   # 1 ts = 1 second 
    1e-3,   # 1 ts = 1 millisecond = 1e3 s
    # 1,      # 1 ts = 1 microsecond = 1e6 s
]

data_repertory = [
    '/home/amelie/Scripts/Data/DVS128Gesture/PLIF_classifier_data/',
    # '/home/amelie/Scripts/Data/NMNIST/NMNIST_classifier_data/',
    # "../data/DVS128Gesture/",
    # "../data/NMNIST/",
]

mut = {
    "mutual": True,
    "sep": False
}

i = 0
for t_ in time_reduction_factors:

    for dr in data_repertory:

        print("\n\n>>",dr)
        if "NMNIST" in dr :
            original_repertory = dr + "subset_data/events_np/"
        else :
            original_repertory = dr+"original_data/events_np/"

        for div in dividers:
            
            for density in densities :

                SNN_repertory = {
                    "mutual": dr+"new_div"+str(div)+"/new_mutualSNN/events_np_"+str(density)+"/",
                    "sep": dr+"new_div"+str(div)+"/new_separateSNN/events_np_"+str(density)+"/"
                }

                for method in ["mutual", "sep"]:
                    init=False
                    nb_events = 0
                    print("\n>>>>>>>>",density, div, method,">>>>>>>>>","\n")

                    for (rep_path, _, files) in walk(original_repertory):
                        repertory=rep_path.replace(original_repertory, "")
                        if len(files) > 0: # and repertory in ["test/0","test/1","train/0","train/1"]:
                            
                            f = 0
                            for event_file in files :
                                print(repertory, event_file)
                                f += 1

                                if not path.exists(join( SNN_repertory[method], repertory , event_file)) and (("NMNIST" in dr and f <= 15) or "DVS128" in dr):
                                
                                    s1 = d.now()
                                    original_events = np.load(path.join(rep_path, event_file))

                                    original_events = np.concatenate((
                                        original_events["x"].reshape(-1,1),
                                        original_events["y"].reshape(-1,1),
                                        original_events["p"].reshape(-1,1),
                                        original_events["t"].reshape(-1,1)
                                    ), axis=1).astype('float64')
                                    nb_events += len(original_events)
                                    # print(original_events[:10])
                                    # print()

                                    while True:
    
                                        # initialize SNN
                                        if init == False:
                                            i = 0
                                            sim.end()
                                            sim.setup(timestep=1)
                                            fullscale_events, downscale_events = init_SNN(sim, original_events, div=div,mutual=mut[method], density=density)
                                            init=True

                                        # SNN
                                        try :
                                            SNN_events = SNN_downscale(
                                                original_events, 
                                                3,
                                                fullscale_events,
                                                downscale_events,
                                                div=div,
                                                keep_polarity=True,
                                                time_reduce=True,
                                                time_factor=t_
                                            )
                                        
                                            if len(SNN_events) == 0:
                                                init = False
                                            else : 
                                                break

                                        except IndexError:
                                            init = False
                                
                                    # SNN_events, SNN_time[method] = computationTime(original_events, fullscale_events, downscale_events, div, SNN_time[method],t_)
                                    saveAsNPZ(SNN_events, join( SNN_repertory[method], repertory ), event_file)

                                    print("total time", d.now() - s1)
                                    print(len(SNN_events))
                                    print()
                                    i += 1