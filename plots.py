import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from reduceEvents import *
from scipy.interpolate import interp1d, PchipInterpolator

P1 = np.array([[ 0.,  0.,  1.,  1.],
    [ 0.,  0.,  1.,  3.],
    [ 0.,  0.,  1.,  4.],
    [ 0.,  0.,  1.,  5.],
    [ 0.,  0.,  0.,  8.],
    [ 0.,  0.,  0., 10.],
    [ 0.,  0.,  1., 12.],
    [ 0.,  0.,  1., 13.],
    [ 0.,  0.,  1., 14.],
    [ 0.,  0.,  1., 16.],
    [ 0.,  0.,  0., 17.],
    [ 0.,  0.,  0., 18.]])
P2 = np.array([[ 0.,  1.,  1.,  2.],
    [ 0.,  1.,  1.,  4.],
    [ 0.,  1.,  1.,  6.],
    [ 0.,  1.,  0.,  9.],
    [ 0.,  1.,  1., 16.],
    [ 0.,  1.,  0., 17.]])
P3 = np.array([[ 1.,  0.,  1.,  2.],
    [ 1.,  0.,  1.,  5.],
    [ 1.,  0.,  0.,  9.],
    [ 1.,  0.,  0., 10.],
    [ 1.,  0.,  1., 14.],
    [ 1.,  0.,  1., 15.],
    [ 1.,  0.,  0., 18.]])
P4 = np.array([[ 1.,  1.,  1.,  2.],
    [ 1.,  1.,  1.,  3.],
    [ 1.,  1.,  1.,  5.],
    [ 1.,  1.,  1.,  6.],
    [ 1.,  1.,  0.,  8.],
    [ 1.,  1.,  0., 11.],
    [ 1.,  1.,  0., 17.],
    [ 1.,  1.,  0., 19.]])

ev = np.vstack((P1,P2,P3,P4))

### SNN

sep_ev, sep_vp, sep_vn = SNN_downscale(ev, 3, div=2, mutual=False, time_reduce=False, plot=True, density=0.05)
mut_ev, mut_vp, mut_vn = SNN_downscale(ev, 3, div=2, mutual=True, time_reduce=False, plot=True, density=0.05)

v_th = -50
v_rest = -65
t = [t/100 for t in range(len(mut_vp))]

# set up
_, ax =plt.subplots()
params = {'mathtext.default': 'regular' } 
for side in ["top","left","right","bottom"]:
    ax.spines[side].set_visible(False)
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")
plt.xlabel("Timestamps", fontsize=12)
plt.ylim(-195,-8)
plt.xlim(-4,max(t)+3)
plt.xticks(range(0,21,2))
ax.set_yticks([])

for e, vp, vn, y_curve, y_re in [(sep_ev, sep_vp, sep_vn, 0,0), (mut_ev, mut_vp, mut_vn, -100, -170)]:
    # voltage
    plt.plot(t,[v + 20 + y_curve for v in vp], color="#82B366")
    plt.plot(t,[v + y_curve for v in vn], color="royalblue")
    plt.hlines([h+y_curve for h in [-65, -50, -45, -30]], 0,max(t)+1, linestyles="--", color="lightgray")
    
    # arrows
    plt.arrow(0, -70 + y_curve, max(t)+1, 0, color="black", head_width=0.5, zorder=3, head_length=0.1)
    plt.arrow(0, -70 + y_curve, 0, 43, color="black", head_width=0.1, zorder=3, head_length=0.7)

    # events
    p, y_min = e, -15
    y_min += y_re
    plt.vlines(p[p[:,2] == 0][:,3], min(-50 + y_curve, y_min), max(-50 + y_curve, y_min), linestyles="--", color="royalblue",  alpha=0.3, zorder=1)
    plt.vlines(p[p[:,2] == 1][:,3], min(-30 + y_curve, y_min), max(-30 + y_curve, y_min), linestyles="--", color="#82B366",  alpha=0.3, zorder=1)
    plt.arrow(0, y_min, max(t)+1, 0, color="black", head_width=0.5, zorder=3, head_length=0.1)
    plt.vlines(p[p[:,2] == 1][:,3], y_min, y_min+5, linewidth=3, color="#82B366", zorder=2)
    plt.vlines(p[p[:,2] == 0][:,3], y_min-5, y_min, linewidth=3, color="royalblue", zorder=2)

# 
done = []
for p, y_min in [
    (P4, -115),
    (P3, -105),
    (P2, -95),
    (P1, -85)
]:
    # if y_min != -15:
    #     plt.vlines([t for t in p[p[:,2] == 0][:,3] if t not in done], -65 + y_curve, -65, linestyles="--", color="lightgray", alpha=0.7, zorder=1)
    #     done += list( p[p[:,2] == 0][:,3] )
    #     plt.vlines([t for t in p[p[:,2] == 1][:,3] if t not in done], -45 + y_curve, -45, linestyles="--", color="lightgray", alpha=0.7, zorder=1)
    #     done += list( p[p[:,2] == 1][:,3] )
    plt.arrow(0, y_min, max(t)+1, 0, color="black", head_width=0.5, head_length=0.1, zorder=3)
    plt.vlines(p[p[:,2] == 1][:,3], y_min, y_min+5, linewidth=3, color="#82B366", zorder=2)
    plt.vlines(p[p[:,2] == 0][:,3], y_min-5, y_min, linewidth=3, color="royalblue", zorder=2)

# rectangle
x1, x2 = [-3, 20.5]
for c,y1,y2 in (['mediumvioletred', -192, -125], ['mediumaquamarine', -75, -8]): 
    plt.vlines([x1,x2], y1,y2, color=c, linewidth=2)
    plt.hlines([y1,y2],x1,x2, color=c, linewidth=2)

# side arrows
plt.arrow(20.5, -82, 0, -36, color="mediumvioletred", head_width=0.15, zorder=3, head_length=0.8)
plt.text(20.9, (-82 -82 - 36)/2, "Mutual\ninfluence", rotation=90, ha="center", va="center", color="mediumvioletred", fontsize=12,weight="bold")
plt.arrow(-0.3, -118, 0, 36, color="mediumaquamarine", head_width=0.15, zorder=3, head_length=0.8)
plt.text(-0.7, (-118 - 118 + 36)/2, "Separate\nhandling", rotation=90, ha="center", va="center", color="mediumaquamarine", fontsize=12,weight="bold")

for e, vp, vn, y_curve, y_re in [(sep_ev, sep_vp, sep_vn, 0,0), (mut_ev, mut_vp, mut_vn, -100, -170)]:    
    for t, y in zip(
        ['Negative reset','Negative threshold', 'Positive reset','Positive threshold', 'Reduced events'],
        [-65 + y_curve, -50 + y_curve, -45 + y_curve, -30 + y_curve, -15 + y_re]):
        plt.text(-0.25, y,t, ha="right", va="center", fontsize=12)


"""
### LOG LUMINANCE

def keepGoodPoints(timestamps,levels, threshold=1):
    ev = np.zeros((0,4))
    T=[]
    L=[]
    old_state = 0
    last_event = None
    for t, new_state in zip(timestamps, levels):
        event = False
        if not new_state % threshold and new_state > old_state :
            if last_event != "neg":
                polarity = 1
                event = True
            last_event = "pos"
        elif not new_state % threshold and new_state < old_state:
            if last_event != "pos":
                polarity = 0
                event = True
            last_event = "neg"
        elif new_state == old_state and old_state != None:
            last_event = (set(["pos", "neg"]) - set([last_event])).pop()
        
        if event: 
            T.append(t)
            L.append(new_state)
            ev = np.vstack((
                ev,
                newEvent(0, 0, polarity, t, 3) 
            ))
        old_state = new_state 
    return T,L, ev


def test(cubic=True):
    X_ = np.linspace(0,20,500)
    Y_ = np.zeros((500,0))
    for i in range(4):
        x,y = np.unravel_index(i, (2,2))
        t, lvl = ev2points(ev[ np.logical_and( ev[:,0]==x , ev[:,1]==y ) ], 1, 2)
        t, lvl = extrapolateLevels(t,lvl,20)
    
        if cubic:
            # cubic_inter = interp1d(t, lvl, kind="cubic", fill_value="extrapolate") # gardé pour illustrer le pb
            cubic_inter = PchipInterpolator(t, lvl)
            Y = np.array(cubic_inter(X_)).reshape(-1,1)
        else : 
            Y = np.interp(X_, t, lvl).reshape(-1,1)
    
        Y_ = np.concatenate(( Y_, Y), axis=1)
        plt.plot(X_,Y, "--")
        plt.scatter(t,lvl,marker="x")
    
    plt.plot(X_, np.mean(Y_, axis=1), color="black")   #moyenne des approximations et non pas approximation de la moyenne
    plt.hlines( range(int(np.max(Y_)) + 2), 0, max(t)+1, linestyles="dotted", color="gray" )
    plt.arrow(0, 0, max(t)+1, 0, color="black", head_width=0.1, zorder=3)
    plt.arrow(0, 0, 0, np.max(Y_)+0.5, color="black", head_width=0.1, zorder=3)
    ax.set_yticklabels([None,None]+list(range(int(np.max(Y_)+2))))
    plt.arrow(0,-1, max(t)+1, 0, color="black", head_width=0.1, zorder=3)

    t,l = extractIntersections(X_,np.mean(Y_, axis=1), 1)
    t,l, p = keepGoodPoints(t,l)
    sc=plt.scatter(t,l,marker="o", color="black",zorder=4)
    sc.set_facecolor("none")
    
    # plot events
    y_min = -1
    plt.vlines(p[p[:,2] == 1][:,3], y_min, y_min+0.4, linewidth=3, color="royalblue", zorder=2)
    plt.vlines(p[p[:,2] == 0][:,3], y_min-0.4, y_min, linewidth=3, color="royalblue", zorder=2)
    plt.vlines(p[:,3], y_min, l, linestyles="dotted", color="gray", zorder=1)

for b in (True, False):
    _, ax = plt.subplots()
    for side in ["top","left","right","bottom"]:
        ax.spines[side].set_visible(False)
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    plt.ylim(-1.5,7)
    plt.xlim(-0.05,22)
    plt.xlabel("Timestamps", fontsize=12)
    plt.ylabel("Log-luminance", rotation=90,fontsize=12, ha="left")
    plt.xticks(range(0,21,2))
    test(cubic=b)
"""
### EVENT COUNT

reduced, EvCount = event_count(ev, 3, div=2, plot=True)
funel = reduce(ev, 3,div=2, temporal=False)

_, ax = plt.subplots()
for side in ["top","left","right","bottom"]:
    ax.spines[side].set_visible(False)
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")
plt.xlabel("Timestamps", fontsize=12)
plt.ylim(-1.5,7)
plt.xlim(-6,23)

# draw arrows
plt.arrow(0, 5.3, 20, 0, color="black", head_width=0.1, zorder=3)
plt.arrow(0, 4, 20, 0, color="black", head_width=0.1, zorder=3)
plt.arrow(0, 3, 20, 0, color="black", head_width=0.1, zorder=3)
plt.arrow(0, 2, 20, 0, color="black", head_width=0.1, zorder=3)
plt.arrow(0, 1, 20, 0, color="black", head_width=0.1, zorder=3)
plt.arrow(0, -0.8, 20, 0, color="black", head_width=0.1, zorder=3)
# side arrows
plt.arrow(20.5, 4.2, 0, -3.4, color="mediumaquamarine", head_width=0.15, zorder=3)
plt.text(20.8, (4.2 + 4.2 - 3.4)/2, "Event Count", rotation=90, ha="center", va="center", color="mediumaquamarine", fontsize=12,weight="bold")
plt.arrow(-0.3, 0.8, 0, 3.4, color="mediumvioletred", head_width=0.15, zorder=3)
plt.text(-0.6, (0.8 + 0.8 + 3.4)/2, "Funnelling", rotation=90, ha="center", va="center", color="mediumvioletred", fontsize=12,weight="bold")

# rectangle
x1, x2 = [-6, 21]
for c,y1,y2 in (['mediumvioletred', 4.6, 6], ['mediumaquamarine', -1.4, 0.45]): 
    plt.vlines([x1,x2], y1,y2, color=c, linewidth=2)
    plt.hlines([y1,y2],x1,x2, color=c, linewidth=2)

# draw events
for p, y_min in [
    (P1, 4),
    (P2, 3),
    (P3, 2),
    (P4, 1),
    (reduced, -0.8),
    (funel, 5.3)
]:
    if y_min != -0.8:
        plt.vlines(p[:,3], 0.3, 5.8, linestyles="--", color="lightgray", zorder=1)
        plt.vlines(p[:,3], -2, 0, linestyles="--", color="lightgray", zorder=1)
    plt.vlines(p[p[:,2] == 1][:,3], y_min, y_min+0.4, linewidth=3, color="royalblue", zorder=2)
    plt.vlines(p[p[:,2] == 0][:,3], y_min-0.4, y_min, linewidth=3, color="royalblue", zorder=2)

# add event count
for t,c in EvCount[0][0]:
    plt.text(t, 0.1, str(c), color="red", fontsize=12, ha="center")

# labels
plt.xticks(range(0,21))
for y, t in zip([-0.8,0.05,1,2,3,4, 5.3], ["Reduced events\nEvent Count", "Normalised event count","Events at pixel 4", "Events at pixel 3", "Events at pixel 2", "Events at pixel 1", "Reduced events\nFunnelling"]):
    plt.text(-1.5, y, t, ha="right", va="center", fontsize=12)
ax.set_yticklabels([])



### LOG LUMINANCE EXPLANATION

# _, ax = plt.subplots()

# # legend
# plt.xlabel("Timestamps")
# plt.xlim(0,20)

# t_P1, lvl_P1 = ev2points(P1, 1, 2)
# plt.hlines(range(floor(min(lvl_P1)), ceil(max(lvl_P1)+1)), 0,20, color="lightgray", linestyle="--",zorder=1)

# t_P1, lvl_P1 = extrapolateLevels(t_P1, lvl_P1, 20)
# cubic_interp = interp1d(t_P1,lvl_P1, kind="cubic")
# T = np.linspace(0,20, 500)
# L = np.array(cubic_interp(T))

# plt.plot(T,L, color="red",zorder=2)
# plt.scatter(t_P1, lvl_P1, marker="x", linewidths=2, zorder=3)

# for 

plt.show()
