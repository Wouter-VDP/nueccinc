import os

folder = "searchingfornues/"
version = "v08_00_00_26"
directory = "/pnfs/uboone/scratch/users/wvdp/"+folder+version
dir_list = os.listdir(directory)

for sample in dir_list:
    print(sample)
    if "run3" in sample:
        outname = sample+".root"
        os.system("hadd -f " + outname + " " + os.path.join(directory,sample) + "/out/*/*.root")
    else:
        outname = "run1_"+sample+".root"
        os.system("hadd -f " + outname + " " + os.path.join(directory,sample) + "/out/*/*.root")
