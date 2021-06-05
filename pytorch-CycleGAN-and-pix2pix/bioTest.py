"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import scanpy as sc
import anndata
import scipy
import torch
from tqdm import tqdm
import numpy as np

def r(x):
    n, embedding_dim = x.shape
    image_dim = (embedding_dim + embedding_dim % 3) // 3
    image_width = int(np.ceil(np.sqrt(image_dim)))
    image_dim = (image_width)**2
    image = torch.zeros((n, image_dim * 3))
    image[:, :embedding_dim] += x
    image = image.reshape(n, 3, image_width, image_width)
    image = torch.nn.functional.pad(input=image, pad=(1, 1, 1, 1))
    return image

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    
    train_adata = anndata.read_h5ad('/dfs/project/CS329D/train_adata_rat.h5ad',)
    print('%d cells, %d genes in training data set.' % train_adata.shape)
    test_adata = anndata.read_h5ad('/dfs/project/CS329D/target_adata_rat.h5ad',)
    print('%d cells, %d genes in training data set.' % test_adata.shape)
    train_data = scipy.sparse.csr_matrix.toarray(train_adata.X)
    test_data = scipy.sparse.csr_matrix.toarray(test_adata.X)
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_data))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = opt.batch_size, shuffle=True, num_workers = 4)
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_data))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.batch_size, shuffle=True, num_workers = 4)

    if opt.eval:
        model.eval()
    for i, data1 in tqdm(enumerate(test_loader)):

        try:
            data2 = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_loader)
            data2 = next(dataloader_iterator)
        data = {}
        data['A'] = r(data1[0])
        data['B'] = r(data2[0])
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
