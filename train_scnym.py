import argparse
import copy
import json
import os
# plotting
import sys
# file downloads
import urllib

import anndata
import numpy as np
import scanpy as sc
from sklearn.metrics import classification_report

sys.path.insert(0, "/afs/cs.stanford.edu/u/prabhat8/cs329d/CS329D/scnym")
from scnym.api import scnym_api, CONFIGS

CONFIGS['no_ssl'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['no_ssl']['ssl_method'] = 'sup_only'
CONFIGS['no_ssl']['ssl_kwargs']['dan_max_weight'] = 0.0
CONFIGS['no_ssl']['ssl_kwargs']['dan_ramp_epochs'] = 1
CONFIGS['no_ssl']['ssl_kwargs']['dan_criterion'] = None
CONFIGS['no_ssl']['unsup_max_weight'] = 0.0
CONFIGS['no_ssl']['description'] = (
    'Train scNym models no SSL'
)

CONFIGS['only_dan'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['only_dan']['mixup_alpha'] = None
CONFIGS['only_dan']['ssl_method'] = 'sup_only'
CONFIGS['only_dan']['unsup_max_weight'] = 0.0
CONFIGS['only_dan']['description'] = (
    'Train scNym models with only domain adversary.'
)

CONFIGS['no_dan'] = copy.deepcopy(CONFIGS['default'])
CONFIGS['no_dan']['ssl_kwargs']['dan_max_weight'] = 0.0
CONFIGS['no_dan']['ssl_kwargs']['dan_ramp_epochs'] = 1
CONFIGS['no_dan']['ssl_kwargs']['dan_criterion'] = None
CONFIGS['no_dan']['description'] = (
    'Train scNym models with MixMatch but no domain adversary. May be useful if class imbalance is very large.'
)

def get_data_tm(basedir, supervised=None):
    train_adata = anndata.read_h5ad('/dfs/project/CS329D/data/mouse_tabula_muris_10x_log1p_cpm.h5ad', )
    target_adata = anndata.read_h5ad('/dfs/project/CS329D/data/mouse_mca_lung_log1p_cpm.h5ad', )

    train_adata.obs['annotations'] = np.array(
        train_adata.obs['cell_ontology_class']
    )
    target_adata.obs['annotations'] = 'Unlabeled'

    adata = train_adata.concatenate(target_adata)

    if supervised == 'target':
        target_adata.obs['annotations'] = np.array(
            target_adata.obs['cell_ontology_class']
        )
        adata = target_adata
    elif supervised == 's+t':
        target_adata.obs['annotations'] = np.array(
            target_adata.obs['cell_ontology_class']
        )
        adata = train_adata.concatenate(target_adata)
    print('%d cells, %d genes in the joined training and target set.' % adata.shape)

    np.savetxt(f'{basedir}/model_genes.csv', train_adata.var_names, fmt='%s')
    model_genes = np.loadtxt(f'{basedir}/model_genes.csv', dtype='str')

    return adata


def get_data_hb(basedir, supervised=None):
    adata = sc.read_h5ad('/dfs/project/CS329D/data/kang_2017_stim_pbmc.h5ad')
    train_adata = adata[adata.obs['stim'] == 'ctrl']
    target_adata = adata[adata.obs['stim'] == 'stim']

    train_adata.obs['annotations'] = np.array(
        train_adata.obs['cell']
    )
    target_adata.obs['annotations'] = 'Unlabeled'

    adata = train_adata.concatenate(target_adata)

    if supervised == 'target':
        target_adata.obs['annotations'] = np.array(
            target_adata.obs['cell']
        )
        adata = target_adata
    elif supervised == 's+t':
        target_adata.obs['annotations'] = np.array(
            target_adata.obs['cell']
        )
        adata = train_adata.concatenate(target_adata)
    # adata = train_adata
    print('%d cells, %d genes in the joined training and target set.' % adata.shape)

    # save genes used in the model
    np.savetxt(f'{basedir}/model_genes.csv', train_adata.var_names, fmt='%s')
    model_genes = np.loadtxt(f'{basedir}/model_genes.csv', dtype='str')

    return adata


def get_data_rat(basedir, supervised=None):
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
    if supervised == 'target':
        target_adata.obs['annotations'] = np.array(
            target_adata.obs['cell_ontology_class']
        )
        adata = target_adata
    elif supervised == 's+t':
        target_adata.obs['annotations'] = np.array(
            target_adata.obs['cell_ontology_class']
        )
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
    elif dataset == 'hb':
        return get_data_hb(path)
    elif dataset == 'hb_target':
        return get_data_hb(path, supervised='target')
    elif dataset == 'rat_target':
        return get_data_rat(path, supervised='target')
    elif dataset == 'hb_st':
        return get_data_hb(path, supervised='s+t')
    elif dataset == 'rat_st':
        return get_data_rat(path, supervised='s+t')
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

    # scnym_tune(
    #     adata=adata,
    #     groupby='annotations',
    #     base_config=config,
    #     out_path=f'{basedir}/scnym_outputs_{config}',
    #     parameters={
    #         "unsup_max_weight": [1.],
    #     },
    #     search="grid",
    #     n_splits=5
    # )


def get_acc(data, key):
    if 'cell_ontology_class' in data.obs:
        true_key = 'cell_ontology_class'
    else:
        true_key = 'cell'


    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_true=np.array(data.obs[true_key].values), y_pred=np.array(data.obs[key].values))
    accuracy = matrix.diagonal() / matrix.sum(axis=1)
    print(accuracy)
    return np.sum(np.array(data.obs[key].values) == np.array(data.obs[true_key].values)) / len(data), accuracy.mean()


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
    print(config, get_acc(adata, key)[0], get_acc(target_adata, key)[0], get_acc(source_data, key)[0], )
    target_adata.obs.to_csv(f'{basedir}/annotations.csv')
    return get_acc(target_adata, key)


def main(args):
    # wandb.init(sync_tensorboard=True, config=args, dir='/dfs/scratch2/prabhat8/wandb')
    accuracies = []
    for i in range(3):
        adata = get_data(os.path.join(args.basedir, 'dataset'), args.dataset)
        train(adata, os.path.join(args.basedir, 'models', args.dataset, str(i)), args.config)
        res = evaluate(os.path.join(args.basedir, 'models', args.dataset, str(i)), adata, args.config)
        accuracies.append(res)
    print("Normal Accuracy: ", np.mean([x[0]*100 for x in accuracies]), np.std([x[0]*100 for x in accuracies]))
    print("Macro Accuracy: ", np.mean([x[1]*100 for x in accuracies]), np.std([x[1]*100 for x in accuracies]))
    os.makedirs(os.path.join(args.basedir, 'results', args.dataset, args.config), exist_ok=True)
    with open(os.path.join(args.basedir, 'results', args.dataset, args.config, 'results.json'), 'w') as f:
        json.dump({'final_result': None, 'accuracies': accuracies}, f)
    # print(final_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--basedir', type=str)
    parser.add_argument('--config', type=str)
    main(parser.parse_args())
