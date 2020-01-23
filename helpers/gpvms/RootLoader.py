import numpy as np
import pandas as pd
import uproot
import awkward
import col_load
import time
import glob
import pickle

### Constants
data_samples = ['on','off']
root_dir = 'nuselection'
main_tree = "NeutrinoSelectionFilter"

### Fiducial volume
lower = np.array([-1.55, -115.53, 0.1])
upper = np.array([254.8, 117.47, 1036.9])
#fid_vol = np.array([[5,6,20], [5,6,50]])
fid_vol = np.array([[10,10,20], [10,10,50]])
fid_box = np.array([lower+fid_vol[0], upper-fid_vol[1]]).T

def is_in_box(x,y,z,box):
    bool_x = (box[0][0] < x) & (x < box[0][1])
    bool_y = (box[1][0] < y) & (y < box[1][1])
    bool_z = (box[2][0] < z) & (z < box[2][1])
    return bool_x & bool_y & bool_z
def is_fid(x,y,z):
    return is_in_box(x,y,z,fid_box)


### Load sample info
def load_sample_info(input_dir, run, exclude_samples):
    start = time.time()
    samples = glob.glob(input_dir+"/*root")
    print(samples)
    temp_names = [k.split('/')[-1].split('.')[0] for k in samples]
    sample_names = ["_".join(k.split("_")[1:]) for k in temp_names]
    print('\nSamples found in directory: ',sample_names)
    print('\nSample Summary: [name, POT, Scaling, Events, SliceID passing rate]')
    sample_dict = dict(zip(sample_names, samples))
    sample_info = {}
    # Exclude samples if wanted
    for k in exclude_samples:
        sample_dict.pop(k, None)
    # Fill POT info 
    if ('on' in sample_dict) and ('off' in sample_dict):
        sample_info['on']={}
        sample_info['off']={}
        data_scaling = pd.read_csv(input_dir + "scaling.txt", index_col=0, sep="\t", header=None).T.iloc[0]
        sample_info['on']['pot'] = data_scaling["tor875_wcut"]
        sample_info['off']['pot'] = 0
        sample_info['on']['scaling'] = 1
        sample_info['off']['scaling'] = data_scaling["E1DCNT_wcut"]/data_scaling["EXT"]    

    for k,v in sample_dict.items():
        file = uproot.open(v)[root_dir]    
        cols_load = col_load.cols_reco.copy()
        fields = [f.decode() for f in file[main_tree].keys()]
        if k not in data_samples:
            sample_info[k] = {}
            cols_load+= (col_load.col_mc+col_load.col_backtracked)
            sample_info[k]['pot'] = file["SubRun"].array("pot").sum()
            if 'on' in sample_info:
                sample_info[k]['scaling'] = sample_info['on']['pot']/sample_info[k]['pot']
            else:
                sample_info[k]['scaling'] = 0
        
        sample_info[k]['fields'] = fields        
        cols_run3_add = [col for col in col_load.cols_run3 if col in fields]
        cols_load+= cols_run3_add
            
        sample_info[k]['daughters'] = file[main_tree].pandas.df(cols_load, flatten=True)
        sample_info[k]['daughters']['trk_min_cos'] = calc_max_angle(file[main_tree])
        sample_info[k]['daughters'].index.names = ['event', 'daughter']
        sample_info[k]['numentries'] = file[main_tree].numentries   
                
        pass_rate = sum(file[main_tree].array("nslice")) / sample_info[k]["numentries"]
        print(k, "\t{:.3g} POT\tScaling: {:.2g}\t{:.0f} events\t NeutrinoID: {:.1f}%".format(sample_info[k]["pot"],
                                                      sample_info[k]["scaling"],
                                                      sample_info[k]["numentries"],
                                                      pass_rate*100))
        duplicates = sum(sample_info[k]["daughters"].xs(0, level="daughter").groupby(by=["evt", "sub", "run",'reco_nu_vtx_z']).size()> 1)
        if duplicates>0:
            print('Duplicated events in sample: {}'.format(duplicates))
        if k not in data_samples:
            mc_arrays, signal_mask_daughters, truth_categories_daughters = load_truth_event(file[main_tree], k)
            sample_info[k]['mc'] = mc_arrays
            sample_info[k]['daughters']['nueccinc'] = signal_mask_daughters
            sample_info[k]['daughters']['true_category'] = truth_categories_daughters
                   
    end = time.time()
    print("\nCompleted, time passed: {:0.1f}s.".format(end - start))
    return sample_info
   

def load_truth_event(tree, name):
    mc_arrays = tree.arrays(col_load.table_cols, namedecode="utf-8")
    mc_arrays["leeweight"] *= mc_arrays["weightSplineTimesTune"]

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
    mc_arrays["true_category"] = truth_categories 
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
  
# Load, Add vars, Pickle!
run=input("Which Run? ")
dir_path = "/uboone/data/users/wvdp/searchingfornues/v08_00_00_33/david_0109/run{}/".format(run)
exclude_samples = []
output = load_sample_info(dir_path, run, exclude_samples)
if input("Do you want to pickle the data? (y/n) ")=="y":
    pickle.dump(output, open(dir_path+"run{}_slimmed.pckl".format(run), "wb"))
    print("Done!")
