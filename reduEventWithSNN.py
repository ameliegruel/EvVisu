"""
Reduce spacially events using 2D convolutional SNN network

Author: Amelie Gruel
Date: 08/2021 - 02/2022
"""

import numpy as np
from math import e, floor, ceil

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
