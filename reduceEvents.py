"""
Function to reduce temporally and spacially the events

Arguments: 
- events (numpy array): events to be reduced
- coord_t (int): index of the timestamp coordinate 
Optionnal arguments:
- div (int): by how much to divide the events (spacial reduction)
- spacial (bool): reduce spacially the events, by dividing the width and height of the captor by the div
- temporal (bool): reduce temporally the events, by rounding thz timestamps to the ms

Author: Amelie Gruel
Date: 08/2021
"""

import numpy as np

def reduce(
    events,
    coord_t,
    div=2,
    spacial=True,
    temporal=True
    ):
    
    if spacial:
        events.T[0] = np.floor(events.T[0]/div)
        events.T[1] = np.floor(events.T[1]/div)
    if temporal:
        events.T[coord_t] = np.floor(events.T[2])
    _, idx=np.unique(events, axis=0, return_index=True)    
    return events[np.sort(idx)]
