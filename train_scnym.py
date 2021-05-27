import argparse
import copy
import json
import os
# file downloads
import urllib

import anndata
import numpy as np
import scanpy as sc
# plotting
from scnym.api import scnym_api, CONFIGS

CONFIGS['no_ssl'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['no_ssl']['ssl_kwargs']['dan_max_weight'] = 0.0
CONFIGS['no_ssl']['ssl_kwargs']['dan_ramp_epochs'] = 1
CONFIGS['no_ssl']['unsup_max_weight'] = 0.0
CONFIGS['no_ssl']['description'] = (
    'Train scNym models with MixMatch but no domain adversary. May be useful if class imbalance is very large.'
)
CONFIGS['only_dan'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['only_dan']['unsup_max_weight'] = 0.0
CONFIGS['only_dan']['description'] = (
    'Train scNym models with only domain adversary. '
)


def get_data_tm(basedir):

    train_adata = anndata.read_h5ad('/dfs/project/CS329D/data/mouse_mca_lung_log1p_cpm.h5ad', )
    target_adata = anndata.read_h5ad('/dfs/project/CS329D/data/mouse_tabula_muris_10x_log1p_cpm.h5ad', )

    train_adata.obs['annotations'] = np.array(
        train_adata.obs['cell_ontology_class']
    )
    target_adata.obs['annotations'] = 'Unlabeled'
    adata = train_adata.concatenate(target_adata)
    print('%d cells, %d genes in the joined training and target set.' % adata.shape)

    np.savetxt(f'{basedir}/model_genes.csv', train_adata.var_names, fmt='%s')
    model_genes = np.loadtxt(f'{basedir}/model_genes.csv', dtype='str')

    return adata


def get_data_rat(basedir):

    cell_atlas_json_url = 'https://storage.googleapis.com/calico-website-scnym-storage/link_tables/cell_atlas.json'
    urllib.request.urlretrieve(
        cell_atlas_json_url,
        './cell_atlas.json'
    )

    with open('./cell_atlas.json', 'r') as f:
        CELL_ATLASES = json.load(f)

    ATLAS2USE = 'rat'

    if ATLAS2USE not in CELL_ATLASES.keys():
        msg = f'{ATLAS2USE} is not available in the cell atlas directory.'
        raise ValueError(msg)

    if not os.path.exists(f'{basedir}/train_data.h5ad'):
        urllib.request.urlretrieve(
            CELL_ATLASES[ATLAS2USE],
            f'{basedir}/train_data.h5ad',
        )
    else:
        print('`train_data.h5ad` is already present.')
        print('Do you really want to redownload it?')
        print('If so, run:')
        print('\t!rm ./train_data.h5ad')
        print('in a cell below.')
        print('Then, rerun this cell.')

    train_adata = anndata.read_h5ad(f'{basedir}/train_data.h5ad', )
    print('%d cells, %d genes in training data set.' % train_adata.shape)

    n_genes = train_adata.shape[1]
    sc.pp.filter_genes(train_adata, min_cells=20)
    n_genes -= train_adata.shape[1]
    print(f'Removed {n_genes} genes.')
    print('%d cells, %d genes in training data set.' % train_adata.shape)

    # save genes used in the model
    np.savetxt(f'{basedir}/model_genes.csv', train_adata.var_names, fmt='%s')

    # temporary
    model_genes = np.loadtxt(f'{basedir}/model_genes.csv', dtype='str')

    # set old cells as target data
    target_adata = train_adata[train_adata.obs['age'] != 'Y', :]
    # use only young cells are training data
    train_adata = train_adata[train_adata.obs['age'] == 'Y', :]

    print('%d cells, %d genes in the training data.' % train_adata.shape)
    print('%d cells, %d genes in the target data.' % target_adata.shape)

    train_adata.obs['annotations'] = np.array(
        train_adata.obs['cell_ontology_class']
    )
    target_adata.obs['annotations'] = 'Unlabeled'

    adata = train_adata.concatenate(target_adata)
    print('%d cells, %d genes in the joined training and target set.' % adata.shape)

    return adata


def get_data(basedir, dataset):
    path = os.path.join(basedir, dataset)
    os.makedirs(path, exist_ok=True)
    if dataset == 'rat':
        return get_data_rat(path)
    elif dataset == 'tm':
        return get_data_tm(path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train(adata, basedir, config):
    scnym_api(
        adata=adata,
        task='train',
        groupby='annotations',
        out_path=f'{basedir}/scnym_outputs_{config}',
        config=config,
    )


def get_acc(data, key):
    return np.sum(np.array(data.obs[key].values) == np.array(data.obs['cell_ontology_class'].values)) / \
           len(data)


def evaluate(basedir, adata, config):
    key = f'{config}_scNym'
    scnym_api(
        adata=adata,
        task='predict',
        key_added=f'{config}_scNym',
        config=config,
        trained_model=f'{basedir}/scnym_outputs_{config}'
    )
    target_adata = adata[adata.obs['annotations'] == 'Unlabeled', :]
    source_data = adata[adata.obs['annotations'] != 'Unlabeled', :]
    print(config, get_acc(adata, key), get_acc(target_adata, key), get_acc(source_data, key), )
    target_adata.obs.to_csv(f'{basedir}/annotations.csv')


def main(args):
    adata = get_data(os.path.join(args.basedir, 'dataset'), args.dataset)
    train(adata, os.path.join(args.basedir, 'models', args.dataset), args.config)
    evaluate(os.path.join(args.basedir, 'models', args.dataset), adata, args.config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--basedir', type=str)
    parser.add_argument('--config', type=str)
    main(parser.parse_args())
