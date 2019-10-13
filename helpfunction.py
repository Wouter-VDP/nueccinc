import numpy as np
import uproot
import awkward
import col_load
import time

### Constants
mass_p = 0.93827
min_p_energy = mass_p + 0.04
min_e_energy = 0.020
mc_samples = ["NUE", "MC", "DRT"]
root_dir = 'nuselection'
main_tree = "NeutrinoSelectionFilter"

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
    start = time.time()
    print('Passing rate after Slice ID:')
    text_file = open(input_dir+file_name, "r")
    sample_info = {}
    for line in text_file.readlines():
        l = line.split()
        if l[0] == "data":
            l[0]="On"
            sample_info["On"] = {}
            sample_info["On"][l[1]] = float(l[3]) * 1e19
            file = uproot.open(input_dir + l[6])[root_dir]
            sample_info["On"]['numentries'] = file[main_tree].numentries
        else:
            sample_info[l[0]] = {}
            sample_info[l[0]][l[3]] = float(l[5])
            file = uproot.open(input_dir + l[7])[root_dir]
            sample_info[l[0]]['numentries'] = file[main_tree].numentries
            if l[0] == "NUE":
                fields = [f.decode() for f in file[main_tree].keys()]
            if l[0] in mc_samples:
                sample_info[l[0]]['POT'] = file["SubRun"].array("pot").sum()
                
        cols_load = col_load.cols_reco
        if l[0] in mc_samples:
            cols_load+= (col_load.col_mc+col_load.col_backtracked)
        sample_info[l[0]]['daughters'] = file[main_tree].pandas.df(cols_load, flatten=True)
        sample_info[l[0]]['daughters']['trk_min_cos'] = calc_max_angle(file[main_tree])
        sample_info[l[0]]['daughters'].index.names = ['event', 'daughter']
    
        pass_rate = sum(file[main_tree].array("n_pfps") > 0) / sample_info[l[0]]["numentries"]
        print(l[0], "\t{:.2f}%".format(pass_rate * 100))
        
        if l[0] in ["MC", "NUE"]:
            sample_info[l[0]]['mc'], sample_info[l[0]]['daughters']['nueccinc'], signal_mask = load_truth_event(file[main_tree], l[0])
            sample_info[l[0]]['signal_mask'] = signal_mask
        
    end = time.time()
    print("Completed, time passed: {:0.1f}s.".format(end - start))
    return sample_info, fields, signal_mask

def load_truth_event(tree, name):
    mc_arrays = tree.arrays(col_load.table_cols, namedecode="utf-8")
    mc_arrays["leeweight"] *= mc_arrays["weightSpline"]

    has_proton = (
        mc_arrays["mc_E"][mc_arrays["mc_pdg"] == 2212] > min_p_energy
    ).any()
    has_electron = (
        mc_arrays["mc_E"][mc_arrays["mc_pdg"] == 11] > min_e_energy
    ).any()
    has_fiducial_vtx = is_fid(
        mc_arrays["true_nu_vtx_x"],
        mc_arrays["true_nu_vtx_y"],
        mc_arrays["true_nu_vtx_z"],
    )

    signal_mask = has_fiducial_vtx & has_electron
    signal_mask_daughters = np.repeat(signal_mask, mc_arrays["n_pfps"])
    pass_rate = sum((signal_mask * mc_arrays["n_pfps"]) > 0) / sum(signal_mask)
    print(name, "sample: nueccinc passing Slice ID \t{:.2f}%".format(pass_rate * 100))
    return mc_arrays, signal_mask_daughters, signal_mask

def calc_max_angle(tree):
    dir_x = tree.array("trk_dir_x_v")
    dir_y = tree.array("trk_dir_y_v")
    dir_z = tree.array("trk_dir_z_v")
    x1, x2 = dir_x.pairs(nested=True).unzip()
    y1, y2 = dir_y.pairs(nested=True).unzip()
    z1, z2 = dir_z.pairs(nested=True).unzip()
    cos_min = ((x1 * x2 + y1 * y2 + z1 * z2) / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2))).min()
    return awkward.topandas(cos_min, flatten=True).clip(upper=1)
    
### Get the pitch
def get_pitch(dir_y, dir_z, plane):
    if plane == 0:
        cos = dir_y * (-np.sqrt(3)/2) + dir_z * (1/2)
    if plane == 1:
        cos = dir_y * (np.sqrt(3)/2) + dir_z * (1/2)
    if plane == 2:
        cos = dir_z
    return 0.3 / cos