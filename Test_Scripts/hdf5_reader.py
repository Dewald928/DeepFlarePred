import h5py
import numpy as np

f = h5py.File('/home/fuzzy/work/Chen_data/SHARP_B_flare_data_300.hdf5', 'r')

print(list(f.keys()))
video = f['video0']
print(list(video.keys()))
frame = video['frame0']
print('done')