# compsci294-predict-IDP

**For running the models and generating samples:**
- VAE_explore.py: Trains variational autoencoder for specific protein and then generates new samples.
- For running the GAN, use this [notebook](https://colab.research.google.com/github/feiglab/idpgan/blob/main/notebooks/idpgan_experiments.ipynb#scrollTo=r_ApGj_IU9hz) and run the section titled "3 - Generate ensembles for a custom protein"
- To setup and run AlphaFlow please follow the instructions in the original [repo](https://github.com/bjing2016/alphaflow/tree/master).
- - To setup the MSA subsampling method please follow the instructions in the original [repo](https://github.com/GMdSilva/gms_natcomms_1705932980_data) and run this [notebook](https://github.com/GMdSilva/gms_natcomms_1705932980_data/blob/main/AlphaFold2_Traj_v1.ipynb). This notebook can be directly run in Google Collaboratory as well.

**For analyzing generated samples:**
- RMSD_analysis.py: Calculates RMSD between every generated structure and every known structure. 
- evaluation_plots.py: Does the calculations for PCA, ROGs, RMSF, and atom clashes, and generates the respective plots reported in Figure 2. of the paper.
- To calculate Wasserstein distances, please follow the instructions and run the [comparison_tool.ipynb](https://gitlab.laas.fr/moma/methods/analysis/WASCO/-/blob/master/wasco/comparison_tool.ipynb?ref_type=heads) in the [WASCO](https://gitlab.laas.fr/moma/methods/analysis/WASCO) repo.