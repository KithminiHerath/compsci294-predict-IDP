import sys
from copy import deepcopy

import biobox as bb
import numpy as np
from tqdm import tqdm

orig_structs = "/Users/claireleblanc/Downloads/PED00159e002_orig_traj.pdb"
generated_structs = "/Users/claireleblanc/Downloads/PED00159e002_GAN_generated_traj_500.pdb"

# Function to get rmsd
def cal_rmsd(input_data, output, test_size):
    test = deepcopy(input_data)
    examine = deepcopy(output)
    re = []
    for j in tqdm(range(0, examine.coordinates.shape[0])):
        test.add_xyz(examine[j])
    for i in tqdm(range(0, input_data.coordinates.shape[0])):
        for j in range(0, output.coordinates.shape[0]):
            val = test.rmsd(i, input_data.coordinates.shape[0] + j)
            # print("({}, {})".format(i, j))
            re.append(val)
    return re

# original conformations
M = bb.Molecule()
# 500 original conformaions --> (4939, 657, 3)
M.import_pdb(orig_structs)
#idx = M.atomselect("*", "*", ["CA", "CB", "C", "N", "CG", "CD", "NE2", "CH3", "O", "OE1"], get_index=True)[1]
idx = M.atomselect("*", "*", ["CA"],get_index=True)[1]
M2 = M.get_subset(idxs=idx)

M1 = bb.Molecule()
# 5000 generated conformations  --> (5000, 832, 3)
M1.import_pdb(generated_structs)
#idx = M1.atomselect("*", "*", ["CA", "CB", "C", "N", "CG", "CD", "NE2", "CH3", "O", "OE1"], get_index=True)[1]
idx = M1.atomselect("*", "*", ["CA"],get_index=True)[1]
M3 = M1.get_subset(idxs=idx)

comp_test = cal_rmsd(M3, M2, test_size=500)
comp_test = np.array(comp_test)
print("> average rmsd: %.3f" % np.mean(comp_test))
np.savetxt("PED000159_GAN_RMSD.csv",comp_test,fmt='%.3f',delimiter=",")
