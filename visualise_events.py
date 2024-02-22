"""
Visualise events following (x,y,t,p)/(x,y,p,t) formalism over time

Tokens: 
- events (.npy or .csv file): events to be visualised, saved using the (x,y,p,t) or (x,y,t,p) formalism
Optionnal tokens:
- unit (string): define the timestamps' unit of measure (either 'ms' or 'nano')
- reduce (bool): reduce the events (using the reduce function from reduceEvents.py)
- region of interest (bool): display the region of interest as a rectangle on the animation 
- save (bool): save the animation 
- save csv as npy (bool): save the converted csv file into a npy file (quicker to process)

Author: Amelie Gruel
Date: 08/2021
"""

from multiprocessing import Event
from timeit import repeat
import numpy as np 
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, ArtistAnimation
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from matplotlib.patches import Rectangle
import argparse
import h5py as h
import sys
import os
from tqdm import tqdm
import math
from datetime import datetime as dt

timestamps = 0
LEN = 60 # length animation in seconds
N = 1

# error handling
def testNumericalInput(user_input):
    while True:
        try : 
            user_input = int(user_input)
            break
        except ValueError :
            user_input = input("Incorrect Value - Please enter a numerical value: ")
    return int(user_input)

def import_data(f_events):

    # read the events
    format_npy=False
    format_csv=False

    if f_events.endswith('npy') or f_events.endswith('npz'):
        events = np.load(f_events)
        format_npy=True
    elif f_events.endswith('csv'):
        events_per_pixel = [list(filter(lambda x: x!='', e)) for e in list(csv.reader(open(f_events,'r'), delimiter=";", quoting=csv.QUOTE_NONNUMERIC))]   
        format_csv=True
    elif f_events.endswith('hdf5'):
        events = h.File(f_events,'r')
        assert 'event' in events.keys()
        events = np.array(events['event'])

    else :
        print("Error : The input file has to be of format .npy, .npz, .hdf5 or .csv")
        sys.exit()

    print("Events correctly loaded from "+f_events+"\n")

    if format_csv:
        events = -1*np.ones((1,4))
        dim=int(np.sqrt(len(events_per_pixel)))
        for x in range(dim):
            for y in range(dim):
                pixel = events_per_pixel[y*dim + x]
                while len(pixel) > 0:
                    events = np.append(events, [[x,y, pixel.pop(), 1]], axis=0)
        if args.save_csv_as_npy:
            np.save(f_events[:-3]+"npy", events)
            print("Processed events correctly saved as "+f_events[:-3]+"npy")
    
    return events

def visualise(f_events, unit="milli", fps=30, size=(None,None), reduce=False, name_events = "datasets", region_of_interest=False, save=False, save_in='./', display=True, accumulation=False, frame=False, first_second=False):

    if save_in[-1] != '/':
        save_in += '/'
    if frame:
        path = save_in+'frames/'
    else:
        path = save_in+'images/'
    
    if type(f_events) == np.ndarray:
        events = f_events
    else:
        events = import_data(f_events)
        name_events = f_events.split('/')[-1]
    
    bool_temp = True

    # adapt to the 3 possible formalisms (x,y,p,t) or (x,y,t,p) or (t,x,y,p)
    string_coord = [None]*4
    set_coord = [2,3]
    if max(events[:,2]) in [0,1]:
        coord_p = 2
        set_coord.remove(2)
    elif max(events[:,3]) in [0,1]:
        coord_p = 3
        set_coord.remove(3)
    string_coord[coord_p] = 'p'

    if max(events[:,0]) > 1e6:
        coord_ts = 0
        coord_x = 1
        coord_y = set_coord[0]
    else:
        coord_ts = set_coord[0]
        coord_x = 0
        coord_y = 1
    string_coord[coord_x] = 'x'
    string_coord[coord_y] = 'y'
    string_coord[coord_ts] = 't'
    
    # get the values for positive and negative polarity
    neg_p = min(events[:,coord_p])
    pos_p = max(events[:,coord_p])

    color_map = {
        neg_p: np.array([30,144,255]), # dodgerblue
        pos_p: np.array([ 0,255,127])  # springgreen
        # neg_p: np.array([ 0,255,127]), # dodgerblue
        # pos_p: np.array([ 0,255,  0])  # springgreen
    }

    # adapt to the timestamps unit of measure
    if unit == "milli":
        frame_interval = 1
        time_length = max(events[:,coord_ts])
        one_second = 1e3
        len_second = LEN*one_second
    elif unit == "nano":
        # frame_interval = 1e6/24 # ms
        time_length = max(events[:,coord_ts])
        one_second = 1e6
        frame_interval = one_second/fps # ms
        len_second = LEN*one_second
    elif unit == "s":
        frame_interval = 0.01
        time_length = max(events[:,coord_ts])
        bool_temp=False
        one_second = 1
        len_second = LEN*one_second

    # get only first second
    if first_second:
        events = events[events[:, coord_ts] < len_second]
        time_length = len_second
    
    # reduce the events
    if reduce : 
        d = testNumericalInput(input("Reduce spatially by: "))
        print("Reducing events...", end=" ")
        events=reduce(events, coord_ts, div=d, temporal=bool_temp)
        print("DONE\n")

    # get width and heigth
    if size == (None,None):
        W = int(max(events[:,coord_x])+1)
        H = int(max(events[:,coord_y])+1)
    else: 
        W = size[0]
        H = size[1]

    # remove empty time at the beginning
    min_t = min(events[:,coord_ts])
    events[:,coord_ts] -= min_t

    # get positive and negative events
    events[:,coord_y] = H - 1 - events[:,coord_y]

    # initialise the figure
    global timestamps
    timestamps=0
    # fig_events = plt.figure(figsize=(10,int(H*10/W)))
    fig_events = plt.figure(figsize=(7,int(H*7/W)))
    ax = plt.axes(xlim=(0, W), ylim=(0,H))

    if accumulation:
        matrix_events = 255*np.ones((H,W,3)).astype(int) # white background
        events = events[events[:,coord_ts] < len_second]
        max_x = max_y = 0
        for ev in tqdm(events):
            if ev[coord_x] > max_x:
                max_x = ev[coord_x]
            if ev[coord_y] > max_y:
                max_y = ev[coord_y]
            matrix_events[int(ev[coord_y]),int(ev[coord_x])] = color_map[ev[coord_p]]
        print(matrix_events)
        print(W,H,'-',max_x, max_y)

        scatter_events = ax.imshow(matrix_events)

        ext = ".png"
        pred = "frame_"

    else : 
        print('Generate video...')

        # in case of foveation, we can display the region of interest
        if region_of_interest : 
            nROI = testNumericalInput(input("How many different regions of interest ? "))
            colors = plt.cm.get_cmap('hsv',nROI)
            for n in range(nROI):
                print(str(n+1)+" region of interest's coordinates:")
                xmin = testNumericalInput(input("x min: "))
                xmax = testNumericalInput(input("x max: "))
                ymin = testNumericalInput(input("y min: "))
                ymax = testNumericalInput(input("y max: "))
                ax.add_patch(Rectangle(
                    (xmin,ymin),
                    xmax-xmin,
                    ymax-ymin,
                    edgecolor=colors(n),
                    fill=False
                ))
                print()

        timestamps = 0
        frames = []
        nb_frames = int(time_length * fps/one_second)

        for f in tqdm(range(nb_frames)):
            # print(f'Step {f}/{nb_frames}') 
            previous_timestamps = timestamps
            timestamps += frame_interval

            frame_events = events[(events[:,coord_ts] >= previous_timestamps) & (events[:,coord_ts] < timestamps)]

            matrix_events = 255 * np.ones((H,W,3),np.int32) # white background  => *0 for black background
            for ev in frame_events:
                matrix_events[int(ev[coord_y]),int(ev[coord_x])] = color_map[ev[coord_p]]
            
            frame = ax.imshow(matrix_events, animated=True)
            if f == 0:
                ax.imshow(matrix_events)
            frames.append([frame])

        animation = ArtistAnimation(fig_events, frames, blit=True, interval=1e3/fps, repeat=True)

        ext = ".gif"
        pred = "animation_"
    
    add=""
    if reduce:
        add=" reduced"
    
    if frame:
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0.002, bottom=0.002, right=0.998, top=0.998)
    else:
        plt.title("Events from "+name_events+add+" over time")
        plt.xlabel("Width (in pixels)")
        plt.ylabel("Height (in pixels)")
    plt.draw()

    # save the video
    if save : 
        print('Saving...')
        if reduce :
            add="reduced_" 
        os.makedirs(path, exist_ok=True)
        f = path+pred+add+name_events.replace('.npy',ext).replace('.hdf5',ext)
        
        if accumulation:
            plt.savefig(f)
            print("Frame correctly saved as "+f)

        else:
            writergif = PillowWriter(fps=fps)
            animation.save(f, writer=writergif)
            # animation.save(f, writer='imagemagick')
            print("Animation correctly saved as "+f)

    if display:
        plt.show()


if len(sys.argv) > 1 :
    start = dt.now()

    # initialise parser
    parser = argparse.ArgumentParser(description="Visualise events over time")
    parser.add_argument("events", metavar="E", type=str, nargs="+", help="Input events with formalism (x,y,t,p)")
    parser.add_argument("--unit", "-u", help="Define the timestamps' unit of measure", nargs=1, type=str, default=["milli"])
    parser.add_argument("--frames-per-second", "-fps", help="Number of frames per second", type=int, default=30)
    parser.add_argument("--accumulation", "-accu", help="Visualise the events accumulated in a frame", action="store_true", default=False)
    parser.add_argument("--reduce", "-r", help="Reduce the events", action="store_true", default=False)
    parser.add_argument("--region_of_interest", "-ROI", help="Visualise the region of interest (in case of foveation)", action="store_true", default=False)
    parser.add_argument("--width", "-W", help="Width of the sensor", type=int, default=None)
    parser.add_argument("--height", "-H", help="Height", type=int, default=None)
    parser.add_argument("--save", "-s", help="Save the animation", action="store_true", default=False)
    parser.add_argument("--save_in", "-si", help="File in which to save the animation", type=str, default="./")
    parser.add_argument("--save_csv_as_npy", "-C2N", help="Save the converted csv file into a npy file (quicker to process)", action="store_true", default=False)
    parser.add_argument("--no-display", "-nd", help="No display of plot", action="store_true", default=False)
    parser.add_argument("--first-second", "-fs", help="Display only first second", action="store_true", default=False)
    args = parser.parse_args()
    
    visualise(
        args.events[0],
        unit=args.unit[0],
        reduce=args.reduce, 
        accumulation=args.accumulation, 
        region_of_interest=args.region_of_interest, 
        size = (args.width, args.height),
        name_events=args.events[0], 
        save=args.save, 
        save_in=args.save_in, 
        display=not args.no_display,
        first_second=args.first_second
    )
    print(f'Time to visualise {args.events[0]}: {(dt.now() - start).total_seconds()} seconds')
