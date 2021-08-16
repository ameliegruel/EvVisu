import os
import tonic 
import tonic.transforms as tr
import numpy as np
import argparse

# initialise parser
parser = argparse.ArgumentParser(description="Download npy files from DVS Gesture dataset")
parser.add_argument("--category", "-c", help="Which category to be downloaded from the dataset", nargs=1, metavar="C", type=int, default=[8])
args = parser.parse_args()

# download and import DVS128 Gesture data
dl = not os.path.isfile('../Bernert_attention/datasets/gesture.zip')
gesture = tonic.datasets.DVSGesture(save_to="./data_gesture", download=dl, train=True)
loader = tonic.datasets.DataLoader(gesture, batch_size=1, shuffle=False)

print("### Data loaded ###\n")
print(args.category[0])

for ev,target in iter(loader):
    print(target.item()+1)
    if target.item()+1 == args.category[0]:
        print("yyayy")
        break

events = ev.numpy()[0]
file_events = "DVS_cat"+str(target.item()+1)+".npy"
np.save(file_events,events)
print("Events correctly saved as "+file_events)