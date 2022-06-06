import argparse
import math
import time

import torch
import torch.nn as nn
from MDSTGNN.net import mdstnet
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import os
from MDSTGNN.util import *
from MDSTGNN.optim import Optim


parser = argparse.ArgumentParser(description='PyTorch Multivariate time series forecasting')
parser.add_argument('--data', type=str, default='./data/exchange_rate.txt.gz',
                    help='location of the data file')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--gl_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--gl_depth', type=int, default=2, help='graph convolution propagation depth')
parser.add_argument('--num_nodes', type=int, default=8, help='number of nodes/variables')
parser.add_argument('--num_layer', type=int, default=3, help='downsampling layers')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--ds_dropout', type=float, default=0.3, help='downsampling dropout rate')
parser.add_argument('--num_neighbors', type=int, default=8, help='k, the number of neighbors of the node')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2)
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--window_length', type=int, default=24*7, help='input sequence length')
parser.add_argument('--out_len', type=int, default=1, help='output sequence length')
parser.add_argument('--horizon', type=int, default=24)
parser.add_argument('--layers', type=int, default=1, help='number of layers')

parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')

parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--prop_beta', type=float, default=0.05, help='prop beta')
parser.add_argument('--tanh_alpha', type=float, default=3, help='tanh alpha')

parser.add_argument('--epochs', type=int, default=30, help='')

parser.add_argument('--step_size', type=int, default=100, help='step_size')

args = parser.parse_args()
device = torch.device(args.device)
torch.manual_seed(111)
np.random.seed(111)
torch.cuda.manual_seed_all(111)
torch.cuda.manual_seed(111)


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X,torch.tensor(range(args.num_nodes),dtype=torch.long).to(device))
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rrmse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rrmse, rae, correlation


def train(data, X, Y, model, criterion, optim, batch_size, scaler, epoch):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    perm = None

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)

        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))

        id = torch.tensor(perm,dtype=torch.long).to(device)
        tx = X[:, :, id, :]
        ty = Y[:, id]
        with autocast():
            output = model(tx, id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:, id]
            loss = criterion(output * scale, ty * scale)

        total_loss += loss.item()
        n_samples += (output.size(0) * data.m)
        scaler.scale(loss).backward()
        grad_norm = optim.step()
        scaler.step(optim.optimizer)
        scaler.update()

        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.5f}'.format(iter, loss.item() / (output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples


def main():
    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.window_length, args.normalize)
    model = None
    if os.path.exists(args.save):
        print('load model')
        with open(args.save, 'rb') as f:
            model = torch.load(f)
    else:
        model = mdstnet(args.gl_true, args.num_layer, args.gl_depth, args.num_nodes,device, args.batch_size,
                        dropout=args.dropout, ds_dropout=args.ds_dropout,number_neighbors=args.num_neighbors,
                        node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                        conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                        skip_channels=args.skip_channels, end_channels=args.end_channels,
                        seq_length=args.window_length, in_dim=args.in_dim, out_dim=args.out_len,
                        layers=args.layers, prop_beta=args.prop_beta, tanh_alpha=args.tanh_alpha,out_len=args.out_len)
    model = model.to(device)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    criterion = nn.L1Loss(reduction='sum').to(device)
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)

    best_val = 100000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )
    scaler = GradScaler()

    try:
        print('start train')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size, scaler,
                               epoch)
            val_rrmse, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                   args.batch_size)
            optim.scheduler.step(val_rrmse)
            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rrmse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_rrmse, val_rae, val_corr), flush=True)

            if val_rrmse < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_rrmse

            test_rrmse, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
            print("test rrmse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_rrmse, test_rae, test_corr), flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    val_rrmse, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
    test_rrmse, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
    print("final test rrmse {:5.4f} |  test rae {:5.4f} | test corr {:5.4f}".format(test_rrmse,  test_rae, test_corr))
    return val_rrmse, val_rae, val_corr, test_rrmse, test_rae, test_corr

if __name__ == "__main__":
    vrrmse = []
    vrae = []
    vcorr = []
    rrmse = []
    rae = []
    corr = []
    for i in range(1):
        val_rrmse, val_rae, val_corr, test_rrmse, test_rae, test_corr = main()
        vrrmse.append(val_rrmse)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        rrmse.append(test_rrmse)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print('10 runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vrrmse), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vrrmse), np.std(vrae), np.std(vcorr)))
    print('\n\n')
    print("test\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(rrmse), np.mean(rae), np.mean(corr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(rrmse), np.std(rae), np.std(corr)))