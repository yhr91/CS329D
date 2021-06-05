# CS329D

## Course Project: Aligning single-cell RNA sequencing data across experiments

###  MixMatch and DANN

To reproduce the results run

```bash
python train_scnym --dataset {dataset} --config {config} --basedir {path to  save models} 
```
The dataset in the command above can be __rat__ for the Rat dataset and __hb__ for the hPMBC dataset.

The config in the above command controls the models as described below:

1. __no_ssl__: Source Only
2. __only_dan__: DANN
3. __no_dan__: MixMatch
4. __no_new_identity__: MixMatch+DANN

The plots can be generated using the __scnym_atlas_transfer.ipynb__ notebook.

### Self-supervised learning

To reproduce results run

```bash
python coarse_trainer.py
python nocoarse_trainer.py
```

To produce plots run the following Jupyter notebook
```
Notebooks/scnym_atlas_transfer-hpbmc-run.ipynb
```
