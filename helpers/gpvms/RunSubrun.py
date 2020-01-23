# Import
import uproot
import numpy as np
import pandas as pd

run=input("Which Run? ")
dir_path = "/uboone/data/users/wvdp/searchingfornues/v08_00_00_33/david_0109/run{}/".format(run)
root_dir = 'nuselection'
tree_name = "SubRun"

fn_on = "run{}_beam_on.root".format(run)
fn_off = "run{}_beam_off.root".format(run)
files = {"On": fn_on, "Off": fn_off}

for name,file in files.items():
    tree=uproot.open(dir_path+file)[root_dir][tree_name]
    out_name = dir_path + "run_subrun_" + name + ".txt"
    print(out_name)
    run_subrun = np.array(tree.arrays(["run", "subRun"], outputtype=pd.DataFrame))
    np.savetxt(out_name, run_subrun, fmt="%d")
