"""
Visualise events following (x,y,t,p)/(x,y,p,t) formalism over time

Tokens: 
- events (.npy or .csv file): events to be visualised, saved using the (x,y,p,t) or (x,y,t,p) formalism
Optionnal tokens:
- unit (string): define the timestamps' unit of measure (either 'ms' or 'nano')
- reduce (bool): reduce the events (using the reduce function from reduceEvents.py)
- save (bool): save the animation 
- temporal (bool): save the converted csv file into a npy file (quicker to process)
- 

Author: Amelie Gruel
Date: 08/2021
"""

import numpy as np 
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import argparse
import sys
from reduceEvents import reduce

# initialise parser
parser = argparse.ArgumentParser(description="Visualise events over time")
parser.add_argument("events", metavar="E", type=str, nargs="+", help="Input events with formalism (x,y,t,p)")
parser.add_argument("--unit", "-u", help="Define the timestamps' unit of measure", nargs=1, type=str, default=["milli"])
parser.add_argument("--reduce", "-r", help="Reduce the events", action="store_true", default=False)
parser.add_argument("--region_of_interest", "-ROI", help="Visualise the region of interest (in case of foveation)", action="store_true", default=False)
parser.add_argument("--save", "-s", help="Save the animation", action="store_true", default=False)
parser.add_argument("--save_csv_as_npy", "-C2N", help="Save the converted csv file into a npy file (quicker to process)", action="store_true", default=False)
args = parser.parse_args()

# read the events
format_npy=False
format_csv=False

try : 
    events = np.load(args.events[0])
    format_npy=True
except (ValueError, FileNotFoundError):
    try :
        events_per_pixel = [list(filter(lambda x: x!='', e)) for e in list(csv.reader(open(args.events[0],'r'), delimiter=";", quoting=csv.QUOTE_NONNUMERIC))]   
        format_csv=True
    except :
        print("Error : The input file has to be of format .npy or .csv")
        sys.exit()

print("Events correctly loaded from "+args.events[0]+"\n")

if format_csv:
    events = -1*np.ones((1,4))
    dim=int(np.sqrt(len(events_per_pixel)))
    for x in range(dim):
        for y in range(dim):
            pixel = events_per_pixel[y*dim + x]
            while len(pixel) > 0:
                events = np.append(events, [[x,y, pixel.pop(), 1]], axis=0)
    if args.save_csv_as_npy:
        np.save(args.events[0][:-3]+"npy", events)
        print("Processed events correctly saved as "+args.events[0][:-3]+"npy")

# adapt to the 2 possible formalisms (x,y,p,t) or (x,y,t,p)
if max(events.T[2] == 1):
    coord_p = 2
    coord_ts = 3
elif max(events.T[3] == 1):
    coord_p = 3
    coord_ts = 2

# get the values for positive and negative polarity
neg_p = min(events.T[coord_p])
pos_p = max(events.T[coord_p])

# adapt to the timestamps unit of measure
if args.unit[0] == "milli":
    frame_interval = 1
    figure_interval = 200
elif args.unit[0] == "nano":
    frame_interval = 1000
    figure_interval = 50


# reduce the events
if args.reduce : 
    print("Reducing events...", end=" ")
    events=reduce(events, coord_ts, div=3)
    print("DONE\n")

# get width and heigth
W = max(events.T[0])
H = max(events.T[1])


# initialise the figure
timestamps=0
fig_events = plt.figure()
ax = plt.axes(xlim=(0,W), ylim=(0,H))
scatter_pos_events = ax.scatter([],[], marker="s", animated=True, color="springgreen", label="Positive events")
scatter_neg_events = ax.scatter([],[], marker="s", animated=True, color="dodgerblue", label="Negative events")

# in case of foveation, we can display the region of interest
if args.region_of_interest : 
    print("Region of interest's coordonnates:")
    xmin = int(input("x min: "))
    xmax = int(input("x max: "))
    ymin = int(input("y min: "))
    ymax = int(input("y max: "))
    ax.add_patch(Rectangle(
        (xmin,ymin),
        xmax-xmin,
        ymax-ymin,
        edgecolor="red",
        fill=False
    ))

# define the animation
def animate(i):
    global timestamps
    previous_timestamps = timestamps
    timestamps += frame_interval
    
    scatter_pos_events.set_offsets(events[(events.T[coord_ts] >= previous_timestamps) & (events.T[coord_ts] < timestamps) & (events.T[coord_p] == pos_p)][: , :2])
    scatter_neg_events.set_offsets(events[(events.T[coord_ts] >= previous_timestamps) & (events.T[coord_ts] < timestamps) & (events.T[coord_p] == neg_p)][: , :2])
    
    return scatter_pos_events, scatter_neg_events,

add=""
animation = FuncAnimation(fig_events, animate, blit=True, interval=figure_interval, save_count=1000)
if args.reduce:
    add=" reduced"
plt.title("Events from "+args.events[0]+add+" over time")
plt.xlabel("Width (in pixels)")
plt.ylabel("Height (in pixels)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.draw()

# save the video
if args.save : 
    if args.reduce :
        add="_reduced" 
    f = "Results/animation_"+args.events[0].split("/")[-1][:-4]+add+".gif" 
    writergif = PillowWriter(fps=frame_interval*1000) 
    animation.save(f, writer=writergif)
    print("Animation correctly saved as "+f)

plt.show()