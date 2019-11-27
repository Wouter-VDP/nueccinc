import numpy as np
import pandas as pd

### Constants
mass_p = 0.93827
min_p_energy = mass_p + 0.04
min_e_energy = 0.020
data_samples = ['on', 'off']

### Fiducial volume
lower = np.array([-1.55, -115.53, 0.1])
upper = np.array([254.8, 117.47, 1036.9])
fid_vol = np.array([[5,6,20], [5,6,50]])
contain_vol = np.array([[10,10,10], [10,10,10]])
fid_box = np.array([lower+fid_vol[0], upper-fid_vol[1]]).T
contain_box = np.array([lower+contain_vol[0], upper-contain_vol[1]]).T
tpc_box = np.array([lower, upper]).T

def is_in_box(x,y,z,box):
    bool_x = (box[0][0] < x) & (x < box[0][1])
    bool_y = (box[1][0] < y) & (y < box[1][1])
    bool_z = (box[2][0] < z) & (z < box[2][1])
    return bool_x & bool_y & bool_z
def is_fid(x,y,z):
    return is_in_box(x,y,z,fid_box)
def is_contain(x,y,z):
    return is_in_box(x,y,z,contain_box)
def is_tpc(x,y,z):
    return is_in_box(x,y,z,tpc_box)
       
### Get the pitch
def get_pitch(dir_y, dir_z, plane):
    if plane == 0:
        cos = dir_y * (-np.sqrt(3)/2) + dir_z * (1/2)
    if plane == 1:
        cos = dir_y * (np.sqrt(3)/2) + dir_z * (1/2)
    if plane == 2:
        cos = dir_z
    return 0.3 / cos
