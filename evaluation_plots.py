import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms, pca
from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)

from Bio import PDB
from tqdm import tqdm

# Define atomic radii for various atom types.
atom_radii = {
#    "H": 1.20,  # Who cares about hydrogen??
    "C": 1.70,
    "CA": 1.70,
    "CB": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "F": 1.47,
    "P": 1.80,
    "CL": 1.75,
    "MG": 1.73,
}

def count_clashes(file, clash_cutoff=0.63):
  orig = mda.Universe(file)

  select = "all"
  clash_cutoffs = {i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j])) for i in atom_radii for j in atom_radii}

  clash_per_struct = []

  for ts in tqdm(orig.trajectory):
    atoms = orig.select_atoms(select)
    coords = np.array(atoms.positions, dtype="d")
    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords) # Structure that makes search easier

    # Initialize a list to hold clashes
    clashes = []
    # Iterate through all atoms
    for i in range(len(coords)):
      # Find atoms that could be clashing
      # print(coords[i])
      atom_1 = atoms[i]
      kdt_search = kdt.search(np.array(coords[i], dtype="d"), max(clash_cutoffs.values()))
      # # Get index and distance of potential clashes
      potential_clash = [(a.index, a.radius) for a in kdt_search]
      # potential_clash = [a.ind for a in kdt_search]

      # Loops through all potential clashes
      for ix, atom_distance in potential_clash:
          atom_2 = atoms[ix]

          # # Exclude clashes from atoms in the same residue
          if atom_1.resid == atom_2.resid:
              continue

          # # Exclude clashes from peptide bonds
          if (atom_2.type == "C" and atom_1.type == "N") or (atom_2.type == "N" and atom_1.type == "C"):
              continue

          # # Exclude clashes from disulphide bridges
          if (atom_2.type == "SG" and atom_1.type == "SG") and atom_distance > 1.88:
              continue

          if atom_2.element + "_" + atom_1.element in clash_cutoffs.keys():
              if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                  clashes.append((atom_1, atom_2))

    clash_per_struct.append(len(clashes) // 2)
  return clash_per_struct


def plot_clash_dist(pdb_file_orig, pdb_files, caption=""):
    '''
        Function to plot the distribution of clash count
        
        Args:
            pdb_file_orig : PDB file path of the trajectory of known conformations
            pdb_files     : List of PDB file paths of the generated trajectories
            caption       : Caption for the figure file to be saved
    '''
    
    orig_clashes = count_clashes(pdb_file_orig)
    
    plt.figure()
    plt.hist(orig_clashes,bins=10,label="Original",alpha=0.5)
    
    # labels = ["VAE", "AlphaFlow", "MSA Subsampling"]
    labels = [file.split('/')[-1][:-4] for file in pdb_files]
    # labels = [file.split('/')[-1][26:-9] for file in pdb_files]
    
    for f in range(len(pdb_files)):
        file = pdb_files[f]
        print(file)
        pred_clashes = count_clashes(file)
        plt.hist(pred_clashes,bins=10,label=labels[f],alpha=0.5)
        
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.ylim(0,1000)
    plt.xlabel("Number of atom clashes")
    plt.ylabel("Count")
    plt.savefig(f'results/clashes{caption}.pdf', dpi=300)
    plt.show()

# Select depends on which atoms have info in pdb file
def make_PCA_plot(pdb_file_orig, pdb_files, select='name CA',caption=""):
    '''
        Function to plot the PCAs of the ensembles
        
        Args:
            pdb_file_orig : PDB file path of the trajectory of known conformations
            pdb_files     : List of PDB file paths of the generated trajectories
            select        : The atoms to select for the PC calculation
            caption       : Caption for the figure file to be saved
    '''
    
    u = mda.Universe(pdb_file_orig)

    # Calculate pc depending on the original structure
    pc = pca.PCA(u, select=select,
             align=True, mean=None,
             n_components=None).run()

    orig_atoms = u.select_atoms(select)
    transformed_orig = pc.transform(orig_atoms, n_components=3)
    df_orig = pd.DataFrame(transformed_orig,columns=['PC{}'.format(i+1) for i in range(3)])
    
    traditional = []
    af2 = []
    
    for file in pdb_files:
        if 'VAE' in file or 'GAN' in file:
            traditional.append(file)
        elif 'alphaflow' in file or 'subsample' in file:
            af2.append(file)
        
    plt.figure()
    plt.scatter(df_orig["PC1"],df_orig['PC2'],s=8,label="Original",alpha=0.5)
    
    # labels = ["VAE", "AlphaFlow", "MSA Subsampling"]
    labels = [file.split('/')[-1][:-4] for file in traditional]
    # labels = [file.split('/')[-1][26:-9] for file in pdb_files]
    
    for f in range(len(traditional)):
        file = traditional[f]
        v = mda.Universe(file)
        new_atoms = v.select_atoms(select)

        # Use PCA to perform dimensionality reduction
        transformed_new = pc.transform(new_atoms, n_components=3)

        df_new = pd.DataFrame(transformed_new,columns=['PC{}'.format(i+1) for i in range(3)])

        if 'VAE' in file:
            color='green'
        elif 'GAN' in file:
            color = 'orange'
        plt.scatter(df_new["PC1"],df_new['PC2'],s=8,label=labels[f],color=color,alpha=0.5) #file.split('/')[-1][17:-9]
        
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Plot")
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f'results/PCA_plot_traditional{caption}.pdf', dpi=300)
    plt.show()
    
    ####
    
    plt.figure()
    plt.scatter(df_orig["PC1"],df_orig['PC2'],s=8,label="Original",alpha=0.5)
    
    # labels = ["VAE", "AlphaFlow", "MSA Subsampling"]
    labels = [file.split('/')[-1][:-4] for file in af2]
    # labels = [file.split('/')[-1][26:-9] for file in pdb_files]
    
    for f in range(len(af2)):
        file = af2[f]
        v = mda.Universe(file)
        new_atoms = v.select_atoms(select)

        # Use PCA to perform dimensionality reduction
        transformed_new = pc.transform(new_atoms, n_components=3)

        df_new = pd.DataFrame(transformed_new,columns=['PC{}'.format(i+1) for i in range(3)])

        if 'subsample' in file:
            color='#84584e'#'brown'
        elif 'alphaflow' in file:
            if 'distilled' in file:
                color = 'red'
            else:
                color = '#8d69b8'
        plt.scatter(df_new["PC1"],df_new['PC2'],s=8,label=labels[f],color=color,alpha=0.5) #file.split('/')[-1][17:-9]
        
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Plot")
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f'results/PCA_plot_af2{caption}.pdf', dpi=300)
    plt.show()
    
# Radius of gyration
def radgyr(atomgroup, masses, total_mass=None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates-center_of_mass)**2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
    sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
    sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses*sq_rs, axis=1)/total_mass
    # square root and return
    return np.sqrt(rog_sq)


def radius_of_gyration_analysis(pdb_file_orig, pdb_files,caption=""):
    '''
        Function to plot the distributions of ROGs
        
        Args:
            pdb_file_orig : PDB file path of the trajectory of known conformations
            pdb_files     : List of PDB file paths of the generated trajectories
            caption       : Caption for the figure file to be saved
    '''

    u = mda.Universe(pdb_file_orig)
    orig_protein = u.select_atoms('protein')
    rog_orig = AnalysisFromFunction(radgyr, u.trajectory,
                            orig_protein, orig_protein.masses,
                            total_mass=np.sum(orig_protein.masses))
    rog_orig.run()
        
    plt.figure()
    plt.hist(rog_orig.results['timeseries'][:,0],bins=50,label="Original",alpha=0.5)
    
    # labels = ["VAE", "AlphaFlow", "MSA Subsampling"]
    labels = [file.split('/')[-1][:-4] for file in pdb_files]
    # labels = [file.split('/')[-1][26:-9] for file in pdb_files]
    
    for f in range(len(pdb_files)):
        file = pdb_files[f]
        v = mda.Universe(file)
        new_protein = v.select_atoms('protein')
        rog_new = AnalysisFromFunction(radgyr, v.trajectory,
                                new_protein, new_protein.masses,
                                total_mass=np.sum(new_protein.masses))
        rog_new.run()
        plt.hist(rog_new.results['timeseries'][:,0],bins=50,label=labels[f],alpha=0.5)
        
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("Radius of Gyration")
    plt.ylabel("Count")
    plt.title("Distributions of RoG")
    plt.savefig(f'results/RoG{caption}.pdf', dpi=300)
    plt.show()
    
def ensembleRMSF(pdb_file, analysis_range, sort=True):
    u = mda.Universe(pdb_file)
    average = align.AverageStructure(u, u, select=analysis_range, ref_frame=0).run() # getting the average structure from the trajectory file.
    # takes the first frame as the reference. aligns all frames to the ref and then calculates the average

    ref = average.results.universe # the above average structure will be the new reference
    aligner = align.AlignTraj(u, ref, select=analysis_range, in_memory=True).run() # aligning each frame "in place" to the reference (averaged structure)
    c_alphas = u.select_atoms(analysis_range)
    R = rms.RMSF(c_alphas).run() # calculating the RMSF on the aligned structures
    return R, c_alphas

def plot_ensembleRMSF(orig_pdb, pdb_files, caption=""):
    '''
        Function to plot the RMSF values of the different trajectories
        
        Args:
            pdb_file_orig : PDB file path of the trajectory of known conformations
            pdb_files     : List of PDB file paths of the generated trajectories
            caption       : Caption for the figure file to be saved
    '''
    
    R, c_alphas = ensembleRMSF(orig_file, "name CA") # backbone
    resid = c_alphas.resids
    first_plotted_residue = min(resid)
    last_plotted_residue = max(resid)
        
    fig, ax = plt.subplots()
    ax.plot(resid, R.results.rmsf, label="Original")
    
    # labels = ["VAE", "AlphaFlow", "MSA Subsampling"]
    labels = [file.split('/')[-1][:-4] for file in pdb_files]
    # labels = [file.split('/')[-1][26:-9] for file in pdb_files]
    
    for f in range(len(pdb_files)):
        file = pdb_files[f]
        R_gen, c_alphas_gen = ensembleRMSF(file, "name CA") # backbone

        ax.plot(resid, R_gen.results.rmsf, label=labels[f])
        
    ax.set_xlabel('Residue number')
    ax.set_ylabel('RMSF ($\AA$)')
    ax.set_title('Ensemble RMSF')
    ax.set_xlim(int(first_plotted_residue), int(last_plotted_residue))
    # plt.legend()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f'results/Ensemble_rmsf{caption}.pdf', dpi=300)
    plt.show()
    

if __name__ == "__main__":
    
    pdb_dir = "predicted_trajectories"

    orig_files = [file for file in sorted(glob.glob(f"{pdb_dir}/*.pdb")) if 'orig_traj' in file]
    
    prots = ['PED00016','PED00159']
    
    for orig_file in orig_files:
        if prots[0] in orig_file:
            prot = prots[0]
        elif prots[1] in orig_file:
            prot = prots[1]
        
        generated_files = [file for file in sorted(glob.glob(f"{pdb_dir}/*.pdb")) if 'orig_traj' not in file and prot in file]

        make_PCA_plot(orig_file, generated_files, select='name CA',caption=f"_{prot}_final")

        radius_of_gyration_analysis(orig_file, generated_files,caption=f"_{prot}_final")

        plot_ensembleRMSF(orig_file, generated_files,caption=f"_{prot}_final")

        plot_clash_dist(orig_file, generated_files, caption=f"_{prot}_final")
        
