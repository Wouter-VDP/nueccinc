import numpy as np
import uproot

### Constants
gr = 1.618


### Fiducial volume
lower = np.array([-1.55, -115.53, 0.1])
upper = np.array([254.8, 117.47, 1036.9])
fid_vol = np.array([[10,10,10], [10,10,50]])
contain_vol = np.array([[10,10,10], [10,10,10]])
fid_box = np.array([lower+fid_vol[0], upper-fid_vol[1]]).T
contain_box = np.array([lower+contain_vol[0], upper-contain_vol[1]]).T

def is_in_box(x,y,z,box):
    bool_x = (box[0][0] < x) & (x < box[0][1])
    bool_y = (box[1][0] < y) & (y < box[1][1])
    bool_z = (box[2][0] < z) & (z < box[2][1])
    return bool_x & bool_y & bool_z
def is_fid(x,y,z):
    return is_in_box(x,y,z,fid_box)
def is_contain(x,y,z):
    return is_in_box(x,y,z,contain_box)


### Load sample info
def load_sample_info(input_dir, file_name):
    text_file = open(input_dir+file_name, "r")
    sample_info = {}
    for line in text_file.readlines():
        l = line.split()
        if l[0] == "data":
            sample_info["On"] = {}
            sample_info["On"][l[1]] = float(l[3]) * 1e19
            sample_info["On"][l[5][:-1]] = uproot.open(input_dir + l[6])['nuselection']
            sample_info["On"]['numentries'] = sample_info["On"]['file']['NeutrinoSelectionFilter'].numentries
        else:
            sample_info[l[0]] = {}
            sample_info[l[0]][l[3]] = float(l[5])
            sample_info[l[0]][l[6][:-1]] = uproot.open(input_dir + l[7])['nuselection']
            sample_info[l[0]]['numentries'] = sample_info[l[0]]['file']['NeutrinoSelectionFilter'].numentries
    return sample_info