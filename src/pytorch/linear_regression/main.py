from __future__ import print_function
import sys
sys.path.append('..')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from approx_mul_pytorch import approx_Conv2d
from approx_mul_pytorch import linear_random_sampling
import time
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch linear regression')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    args = parser.parse_args()
    #use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = False
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    sample_ratio = args.sample_ratio
    print("using device " + str(device))
    print("sample ratio {}".format(sample_ratio))    
    n_train = 1000 #number of training examples
    n_test = 1000 #test set examples
    dim = 5
    real_w = torch.rand(dim, 1)
    x_train = torch.rand(dim, n_train)
    y_train = torch.mm(real_w.t(), x_train) 
    x_test = torch.rand(dim, n_train)
    y_test = torch.mm(real_w.t(), x_test)
    eta = 0.0001
    iter = 10000
    minimal_k = 1
    w = torch.zeros(dim,1,requires_grad=True) + 1e-5
    for _ in range(iter):
        #y = torch.mm(w.t(), x_train)
        y = linear_random_sampling(w.t(),x_train,None,sample_ratio, minimal_k, with_replacement=True, optimal_prob=False, scale=True,sample_ratio_bwd=None, minimal_k_bwd=None, sample_ratio_wu=None, minimal_k_wu=None)
        #l = (y-y_test).norm(p=2)/n_train 
        l = torch.mm((y-y_test),(y-y_test).t())
        w.retain_grad()
        l.backward(retain_graph=True)
        #print(w.grad)
        #print('MSE: {}'.format(l))
        #print('real w: {}'.format(real_w))
        #print('w: {}'.format(w))
        grad_w = 2*torch.mm(x_train,(y-y_train).t())
        grad_w = w.grad
        w = w - eta*grad_w
        #w.grad.data.zero_()
    
    print('real w: {}'.format(real_w))
    print('w: {}'.format(w))
    y = torch.mm(w.t(), x_train)
    print('train loss: {}'.format((y-y_train).norm())) 
    y = torch.mm(w.t(), x_test)
    print('test loss: {}'.format((y-y_test).norm())) 


if __name__ == '__main__':
    main()
