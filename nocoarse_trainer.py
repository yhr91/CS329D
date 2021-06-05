# allow tensorboard outputs even though TF2 is installed
# broke the tensorboard/pytorch API
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
#import scnym

import sys
sys.path.append('../')
from datetime import datetime

# import scnym
from scnym_orig import scnym
from scnym_orig.scnym.api import scnym_api
import torch

# file downloads
import urllib
import json
import os

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

name = 'raw'

def decode_bytes_h5ad(adata):

    try:
        adata.var_names = [x.decode('utf-8') for x in adata.var_names]
        adata.obs.index = [x.decode('utf-8') for x in adata.obs.index]
    except:
        pass

    ## Everything gets read in as bytes, so we decode each column in the dataframe here
    for c in adata.obs.columns:
        try:
            adata.obs[c] = [x.decode('utf-8') for x in adata.obs[c]]
        except:
            continue
    return adata

for it in range(5):

	adata = sc.read_h5ad('/dfs/user/yhr/CS329D/training_sets/'+name+'.h5ad')
	adata = decode_bytes_h5ad(adata)

	augment = []

	train_adata = adata[adata.obs['stim'] == 'ctrl']
	target_adata = adata[adata.obs['stim'] == 'stim']

	for c in augment:
	    train_adata = augment_class(train_adata, c)
	    target_adata = augment_class(target_adata, c)

	train_adata.obs['annotations'] = np.array(
	    train_adata.obs['cell']
	)
	target_adata.obs['annotations'] = 'Unlabeled'

	adata = train_adata.concatenate(target_adata)

	print('%d cells, %d genes in the joined training and target set.' % adata.shape)

	# save genes used in the model
	np.savetxt('./model_genes.csv', train_adata.var_names, fmt='%s')
	model_genes = np.loadtxt('./model_genes.csv', dtype='str')

	dt = str(datetime.now())[5:19].replace(' ', '_').replace(':', '-')
	model_name = '/dfs/user/yhr/CS329D/training_sets/results_map_dict_'+name+'_'+dt
	print('Training '+model_name)

	scnym_api(
	    adata=adata,
	    task='train',
	    groupby='annotations',
	    out_path='./scnym_outputs/' + name,
	    config='no_ssl',
	)

	#subset_adata = adata[adata.obs['augmented'] != True]

	scnym_api(
	    adata=target_adata,
	    task='predict',
	    key_added='scNym',
	    config='no_new_identity',
	    trained_model='./scnym_outputs/' + name,
	)

	target_adata.write_h5ad(model_name+'.h5ad')
