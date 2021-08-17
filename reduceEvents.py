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

    def reduce_1event(event):
        x=event
        if spacial:
            x[0] = event[0]/div
            x[1] = event[1]/div 
        if temporal:
            x[coord_t] = int(round(event[coord_t]))
        return x

    reduced_events = np.array(list(map(reduce_1event, events)))
    
    return reduced_events
