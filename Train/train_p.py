import os
import datetime
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Model.CNN_Transformer import TransformerNN as TransformerNN
from TensorProcessing.Dataset import FourWellDataset
from Model.FDM_filters import *
import torch
import torch.nn as nn
from torch import linalg as LA
import time
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset


def set_default():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.autograd.set_detect_anomaly(False)
    torch.manual_seed(3407)
    torch.set_default_dtype(torch.float32)


def train_loop(dataset_loader, gpu, net, optimizer, loss_criterion):
    net.train()
    num_batch = len(dataset_loader)
    total_loss_all = 0

    print(f'num of batch = {num_batch}')

    for batch_idx, (X1, X2, Y) in enumerate(dataset_loader):
        # X1 shape: (b, 2, 6, 50, 50), X2 shape: (b, 84, 1), Y shape: (b, 84, 6, 50, 50)

        X1 = X1.to(gpu)
        X2 = X2.to(gpu)
        Y = Y.to(gpu)

        pred, _ = net(X1, X2)

        loss = loss_criterion(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_all += loss.item()

        # if batch_idx % 50 == 0:
        #     loss_value_all = loss.item()
        #     loss_value_l2, current_batch = loss_l2.item(), batch_idx * len(Y)
        #     print(f'Loss all: {loss_value_all:>15f} | Loss L2: {loss_value_l2:>15f} | [{current_batch:>5d}/{size:>5d}]')

    print(f'&& Training Error: avg loss all = {total_loss_all/num_batch:.5e}')
    return total_loss_all/num_batch


def val_loop(dataloader, gpu, net, loss_criterion):
    net.eval()
    num_batches = len(dataloader)
    val_loss_all = 0

    with torch.no_grad():
        for batch, (X1, X2, y) in enumerate(dataloader):
            X1, X2, y = X1.to(gpu), X2.to(gpu), y.to(gpu)
            pred, list_of_weights = net(X1, X2)

            loss = loss_criterion(pred, y)

            val_loss_all += loss.item()

    val_loss_all /= num_batches
    print(f'&& Validation Error: avg loss all = {val_loss_all:.5e}')
    # print(f'Attention weights: {list_of_weights[-1][0, 0, :, :]}')
    return val_loss_all


if __name__ == '__main__':

    set_default()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, GPU name: {torch.cuda.get_device_name()}')

    state_variable = 'pressure'
    train_dataset = FourWellDataset(train_or_val='train', label_name=state_variable)
    val_dataset = FourWellDataset(train_or_val='val', label_name=state_variable)
    batch_size = 10

    print(f'Len of trainset={len(train_dataset)}, len of valset={len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)
    print(f'Data loading finished.')

    model = TransformerNN(
        transformer_embed_size=200,
        transformer_target_size=200,
        transformer_num_layers=2,
        transformer_heads=8,
        transformer_forward_expansion=4,
        transformer_dropout=0.0,
        transformer_dt=86400 * 30 * 6,
        transformer_num_ts=40,
        device='cuda',
        geo_in_channel=8,
        geo_embed_size=200,
        rate_embed_size=200,
        decoder_channel=40
    ).to(device)

    print(f'Model total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    learning_rate = 1e-3
    adam_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05,)  # 0.05
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(adam_optimizer, mode='min', factor=0.1, patience=20, threshold=0.001,
                                  threshold_mode='rel', cooldown=0, min_lr=1.e-8, eps=1e-08)  # 40 1e-8
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(adam_optimizer, 15, 2, eta_min=1e-9)

    epochs = 800

    train_loss_list_all = []
    val_loss_list_all = []

    begins = time.time()
    for tt in range(epochs):
        print(f'Epoch {tt + 1}\n===========================================')
        begin1 = time.time()
        train_all_loss_epoch_all = train_loop(train_loader, device, model, adam_optimizer, criterion)
        val_all_loss_epoch = val_loop(val_loader, device, model, criterion)

        scheduler.step(val_all_loss_epoch)

        train_loss_list_all.append(train_all_loss_epoch_all)
        val_loss_list_all.append(val_all_loss_epoch)

        end1 = time.time()
        print(f'current learning rate: {adam_optimizer.param_groups[0]["lr"]}')
        print(f'******* This epoch takes {(end1 - begin1):.3f} s. *******')
        print(f'******* All epochs takes {(end1 - begins) / 60:.2f} min. *******\n')

    save_name = 'CNN-Transformer_baseline'

    torch.save(model.state_dict(),
               f'G:\\optim_code\\checkpoints\\{state_variable}_{save_name}.pth')
    np.savetxt(f'G:\\optim_code\\losses\\training_loss_{state_variable}_{save_name}.txt', train_loss_list_all)
    np.savetxt(f'G:\\optim_code\\losses\\validation_loss_{state_variable}_{save_name}.txt', val_loss_list_all)

    plt.semilogy(np.arange(len(train_loss_list_all)), train_loss_list_all, ls='-', c='b', label='training loss')
    plt.semilogy(np.arange(len(val_loss_list_all)), val_loss_list_all, ls='-', c='r', label='validating loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'G:\\optim_code\\plots\\{state_variable}_{save_name}.png', dpi=300)
    plt.close()

    print(f'\n========= Every epochs takes {(time.time() - begins) / epochs:.4f} s. =========')
    print('=========Done===========')

