# compsci294-predict-IDP

**For running the models and generating samples:**
- VAE_explore.py: Trains variational autoencoder for specific protein and then generates new samples.
- For running the GAN, use this [notebook](https://colab.research.google.com/github/feiglab/idpgan/blob/main/notebooks/idpgan_experiments.ipynb#scrollTo=r_ApGj_IU9hz) and run the section titled "3 - Generate ensembles for a custom protein"

**For analyzing generated samples:**
- RMSD_analysis.py: Calculates RMSD between every generated structure and every known structure. 
