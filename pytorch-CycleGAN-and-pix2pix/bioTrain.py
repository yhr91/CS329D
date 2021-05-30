"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
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
    image = np.zeros((n, image_dim * 3))
    image[:, :embedding_dim] = x
    image = image.reshape(n, 3, image_width, image_width)
    return image

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
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

    dataset_size = len(test_data)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        dataloader_iterator = iter(train_loader)

        for i, data1 in tqdm(enumerate(test_loader)):

            try:
                data2 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_loader)
                data2 = next(dataloader_iterator)
            data = {}
            data['A'] = r(data1[0])
            data['B'] = r(data2[0])

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
