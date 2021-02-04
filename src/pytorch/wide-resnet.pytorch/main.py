from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from networks import *
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--sample_ratio', default=0.5, type=float, help='sample ratio for approx conv2d')
parser.add_argument('--minimal_k', default=1, type=int, help='minimal_k for approx conv2d')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, args.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, args.sample_ratio, args.minimal_k)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct.float()/total
    print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


def get_approx_layers_aux(model, layers):
    num = 0
    if model.__class__.__name__ == 'approx_Conv2d':
        for n,p in model.named_parameters():
            #print(n, p.size())
            if ('weight' in n):
                #num += 1
                layers.append(model)
    for layer in model.children():
        get_approx_layers_aux(layer,layers)
    return layers

def get_approx_layers(model):
    approx_layers = []
    return get_approx_layers_aux(model,approx_layers)

# Training
def train(epoch):
    compare_grad_vs_approx = False
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
    if compare_grad_vs_approx == True:
        num_approx_layers = len(get_approx_layers(net))
        avg_mean = torch.zeros(num_approx_layers, len(trainloader))
        avg_mse = torch.zeros(num_approx_layers, len(trainloader))
        avg_std = torch.zeros(num_approx_layers, len(trainloader))
        max_diff = torch.zeros(num_approx_layers, len(trainloader))

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        
        if compare_grad_vs_approx == True:
            # get gradients with non-approximate calculations:
            acc_grads = []
            for layer in get_approx_layers(net):
                layer.eval() 
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)               # Forward Propagation
            loss = criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            for layer in get_approx_layers(net):
                for n,p in layer.named_parameters():
                    if ('weight' in n):
                        acc_grads.append(p.grad.clone())
            
            approx_grads = []
            net.train()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)               # Forward Propagation
            loss = criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            for layer in get_approx_layers(net):
                for n,p in layer.named_parameters():
                    if ('weight' in n):
                        approx_grads.append(p.grad.clone())
            
            optimizer.step() # Optimizer update
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            sys.stdout.write('\n')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.2f%%'
                    %(epoch, num_epochs, batch_idx+1,
                        (len(trainset)//batch_size)+1, loss.item(), 100.*correct.float()/total))
            
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
            sys.stdout.flush()
        
        else:
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)               # Forward Propagation
            loss = criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            sys.stdout.write('\n')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.2f%%'
                    %(epoch, num_epochs, batch_idx+1,
                        (len(trainset)//batch_size)+1, loss.item(), 100.*correct.float()/total))
            sys.stdout.flush()
            #print(batch_idx)
            #print(prof.key_averages())
            #exit()
            #print(net.module.layer1[1].conv1.weight[0][0])
    
    if compare_grad_vs_approx == True:
        torch.set_printoptions(linewidth=100000)
        print()
        print('mean {}'.format(torch.mean(avg_mean,dim=1)))
        print('relative MSE {}'.format(torch.mean(avg_mse,dim=1)))
        print('std {}'.format(torch.mean(avg_std,dim=1)))
        print('max diff {}'.format(torch.norm(max_diff,p=float('inf'),dim=1)))
        print('avg max diff {}'.format(torch.mean(max_diff,dim=1)))
        torch.set_printoptions(profile="default")

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct.float()/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net.module if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
