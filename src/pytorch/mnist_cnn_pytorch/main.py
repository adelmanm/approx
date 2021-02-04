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
import time
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

t_start = None
train_size = None
validation_size = None

sample_ratio = 0.5
minimal_k = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv1 = approx_Conv2d(1, 32, kernel_size=5, padding=2, sample_ratio=sample_ratio, minimal_k=minimal_k)
        self.conv2 = approx_Conv2d(32, 64, kernel_size=5, padding=2, sample_ratio=sample_ratio, minimal_k=minimal_k)
        self.conv2_drop = nn.Dropout2d()
        #disable dropout for gradient experiemnts
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #disable dropout for gradient experiemnts
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        #disable dropout for gradient experiemnts
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    compare_grad_vs_approx = False
    model.train()
    
    if compare_grad_vs_approx == True:
        num_approx_layers = 0
        for layer in model.children():
            #print(layer.__class__.__name__)
            if layer.__class__.__name__ == 'approx_Conv2d':
                for n,p in layer.named_parameters():
                    #print(n, p.size())
                    if ('weight' in n):
                        num_approx_layers += 1
        
        
        avg_mean = torch.zeros(num_approx_layers, len(train_loader))
        avg_mse = torch.zeros(num_approx_layers, len(train_loader))
        avg_std = torch.zeros(num_approx_layers, len(train_loader))
        max_diff = torch.zeros(num_approx_layers, len(train_loader))



    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if compare_grad_vs_approx == True:
            # get gradients with non-approximate calculations:
            acc_grads = []
            for layer in model.children():
                if layer.__class__.__name__ == 'approx_Conv2d':
                    layer.eval() 
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            for layer in model.children():
                if layer.__class__.__name__ == 'approx_Conv2d':
                    for n,p in layer.named_parameters():
                        if ('weight' in n):
                            acc_grads.append(p.grad.clone())
                    
            
            approx_grads = []
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            for layer in model.children():
                if layer.__class__.__name__ == 'approx_Conv2d':
                    for n,p in layer.named_parameters():
                        if ('weight' in n):
                            approx_grads.append(p.grad.clone())
            
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} elapsed_time: {:.2f} sec'.format(
                    epoch, batch_idx * len(data), train_size,
                    100. * batch_idx / len(train_loader), loss.item(), time.time()-t_start))
        
            #print('approx_grads:')
            #print (approx_grads)
            #print ('acc_grads') 
            #print (acc_grads) 
            #print('mean {}'.format(torch.mean(avg_mean,dim=1)))
            #print('relative MSE {}'.format(torch.mean(avg_mse,dim=1)))
            #print('std {}'.format(torch.mean(avg_std,dim=1)))
            for i, (approx_grad,acc_grad) in enumerate(zip(approx_grads,acc_grads)):
                #print('index {}'.format(i))
                #print('mean {}'.format((approx_grad-acc_grad).flatten().mean()))
                avg_mean[i,batch_idx] = (approx_grad-acc_grad).flatten().mean()
                #print('relative MSE {}'.format((approx_grad-acc_grad).norm()/acc_grad.norm()))
                avg_mse[i,batch_idx] = (approx_grad-acc_grad).norm()/acc_grad.norm()
                #print('std {}'.format((approx_grad-acc_grad).flatten().std()))
                avg_std[i,batch_idx] = (approx_grad-acc_grad).flatten().std()
                #print('max diff {}'.format((approx_grad-acc_grad).norm(p=float('inf'))))
                max_diff[i,batch_idx] = (approx_grad-acc_grad).norm(p=float('inf'))
        
        else:
            optimizer.zero_grad()
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} elapsed_time: {:.2f} sec'.format(
                    epoch, batch_idx * len(data), train_size,
                    100. * batch_idx / len(train_loader), loss.item(), time.time()-t_start))
            #print(prof)
            #print(prof.key_averages())
            #exit()
    
    if compare_grad_vs_approx == True:
        print('mean {}'.format(torch.mean(avg_mean,dim=1)))
        print('relative MSE {}'.format(torch.mean(avg_mse,dim=1)))
        print('std {}'.format(torch.mean(avg_std,dim=1)))
        print('max diff {}'.format(torch.norm(max_diff,p=float('inf'),dim=1)))
        print('avg max diff {}'.format(torch.mean(max_diff,dim=1)))

def validate(args, model, device, validation_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) elapsed_time: {:.2f} sec\n'.format(
        validation_loss, correct, validation_size,
        100. * correct / validation_size, time.time()-t_start))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--validation_size', type=int, default=5000, metavar='N',
                        help='how many images to use as validation set (default: 5000)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print("using device " + str(device))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    mnist_train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    mnist_test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    # Define the indices
    indices = list(range(len(mnist_train_dataset)))  # start with all the indices in training set

    global train_size, validation_size
    validation_size = args.validation_size  # define the split size
    train_size = len(mnist_train_dataset) - validation_size
    print('training set size: {} samples',train_size)
    print('validation set size: {} samples', validation_size)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=validation_size, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset = mnist_train_dataset,
        batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        dataset=mnist_train_dataset,
        batch_size=args.validation_size, sampler=validation_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset = mnist_test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    global t_start
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        validate(args, model, device, validation_loader)
    t_end = time.time()
    print('Total training time: {:.2f} sec'.format(t_end-t_start))
    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
