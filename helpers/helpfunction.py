import numpy as np
import pandas as pd
import uproot
import awkward
from helpers import col_load
import time
import glob

### Constants
mass_p = 0.93827
min_p_energy = mass_p + 0.04
min_e_energy = 0.020
data_samples = ['on','off']
root_dir = 'nuselection'
main_tree = "NeutrinoSelectionFilter"

### Fiducial volume
lower = np.array([-1.55, -115.53, 0.1])
upper = np.array([254.8, 117.47, 1036.9])
fid_vol = np.array([[5,6,20], [5,6,50]])
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
def load_sample_info(input_dir, run, exclude_samples):
    start = time.time()
    print('Sample Summary: [name, POT, Scaling, Events, SliceID passing rate]')
    samples = glob.glob(input_dir+"/*root")
    sample_names = ["_".join(k.split("_")[2:-1]) for k in samples]
    sample_dict = dict(zip(sample_names, samples))
    sample_info = {}
    # Exclude samples if wanted
    for k in exclude_samples:
        sample_dict.pop(k, None)
    # Fill POT info 
    if ('on' in sample_dict) and ('off' in sample_dict):
        sample_info['on']={}
        sample_info['off']={}
        data_scaling = pd.read_csv(input_dir + "scaling.txt", index_col=0, sep=" ", header=None).T.iloc[0]
        sample_info['on']['pot'] = data_scaling["tor875_wcut"]
        sample_info['off']['pot'] = 0
        sample_info['on']['scaling'] = 1
        sample_info['off']['scaling'] = data_scaling["E1DCNT_wcut"]/data_scaling["EXT"]    

    for k,v in sample_dict.items():
        file = uproot.open(v)[root_dir]    
        cols_load = col_load.cols_reco.copy()
        if k not in data_samples:
            sample_info[k] = {}
            cols_load+= (col_load.col_mc+col_load.col_backtracked)
            sample_info[k]['pot'] = file["SubRun"].array("pot").sum()
            sample_info[k]['scaling'] = sample_info['on']['pot']/sample_info[k]['pot']
        if run==3:
            cols_load+= cols_run3
            
        sample_info[k]['daughters'] = file[main_tree].pandas.df(cols_load, flatten=True)
        sample_info[k]['daughters']['trk_min_cos'] = calc_max_angle(file[main_tree])
        sample_info[k]['daughters'].index.names = ['event', 'daughter']
        sample_info[k]['numentries'] = file[main_tree].numentries   
        
        pass_rate = sum(file[main_tree].array("nslice")) / sample_info[k]["numentries"]
        print(k, "\t{:.3g} POT\tScaling: {:.2g}\t{:.0f} events\t NeutrinoID: {:.1f}%".format(sample_info[k]["pot"],
                                                      sample_info[k]["scaling"],
                                                      sample_info[k]["numentries"],
                                                      pass_rate*100))
        
        if k not in data_samples:
            mc_arrays, signal_mask_daughters, truth_categories_daughters = load_truth_event(file[main_tree], k)
            sample_info[k]['mc'] = mc_arrays
            sample_info[k]['daughters']['nueccinc'] = signal_mask_daughters
            sample_info[k]['daughters']['truth_cat'] = truth_categories_daughters
            if k == "nue":
                fields = [f.decode() for f in file[main_tree].keys()]
                   
    end = time.time()
    print("Completed, time passed: {:0.1f}s.".format(end - start))
    return sample_info, fields
   

def load_sample_info_old(input_dir, file_name):
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
        print(l[0], "\t{:.2f}%\t [{:.0f} events]".format(pass_rate * 100, sample_info[l[0]]["numentries"]))
        
        if l[0] in ["MC", "NUE", "NC"]:
            mc_arrays, signal_mask_daughters, signal_mask, truth_categories, truth_categories_daughters = load_truth_event(file[main_tree], l[0])
            sample_info[l[0]]['mc'] = mc_arrays
            sample_info[l[0]]['daughters']['nueccinc'] = signal_mask_daughters
            sample_info[l[0]]['nueccinc'] = signal_mask
            sample_info[l[0]]['daughters']['truth_cat'] = truth_categories_daughters
            sample_info[l[0]]['truth_cat'] = truth_categories
        
    end = time.time()
    print("Completed, time passed: {:0.1f}s.".format(end - start))
    return sample_info, fields

def load_truth_event(tree, name):
    mc_arrays = tree.arrays(col_load.table_cols, namedecode="utf-8")
    mc_arrays["leeweight"] *= mc_arrays["weightSpline"]

    has_fiducial_vtx = is_fid(
        mc_arrays["true_nu_vtx_x"],
        mc_arrays["true_nu_vtx_y"],
        mc_arrays["true_nu_vtx_z"],
    )
    has_proton = mc_arrays['nproton']>0
    has_electron = mc_arrays['nelec']>0
    has_pion = (mc_arrays['npi0'] + mc_arrays['npion'])>0 
    has_muon = mc_arrays['nmuon']>0

    signal_mask = has_fiducial_vtx & has_electron
    mc_arrays["nueccinc"] = signal_mask
    signal_mask_daughters = np.repeat(signal_mask, mc_arrays["n_pfps"])
    if sum(signal_mask):
        pass_rate = sum((signal_mask * mc_arrays["n_pfps"]) > 0) / sum(signal_mask)
        print(name, "sample: nueccinc passing Slice ID \t{:.2f}%".format(pass_rate * 100))
    
    nuecc0p0pi = signal_mask & ~has_proton & ~has_pion & ~has_muon 
    nueccNp0pi = signal_mask & has_proton & ~has_pion & ~has_muon
    nueccNpMpi = signal_mask & has_pion & ~has_muon
    nunc = has_fiducial_vtx & ~has_electron & ~has_muon
    numucc = has_fiducial_vtx & has_muon
    nu_out_of_fv = ~has_fiducial_vtx
    truth_categories = nuecc0p0pi + 2*nueccNp0pi + 3*nueccNpMpi + 4*nunc + 5*numucc + 6*nu_out_of_fv
    mc_arrays["true_category"] = signal_mask 
    truth_categories_daughters = np.repeat(truth_categories, mc_arrays["n_pfps"])
    
    return mc_arrays, signal_mask_daughters, truth_categories_daughters

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