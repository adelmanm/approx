# Faster Neural Network Training with Approximate Tensor Operations

This repository is the official implementation of [Faster Neural Network Training with Approximate Tensor Operations](https://arxiv.org/abs/1805.08079) (NeurIPS 2021). 
![fig_sample_conv](https://user-images.githubusercontent.com/18640225/137972406-9f759402-e8a1-4715-b85a-33258c8dbb9e.png)

## Requirements

The code in this repository was ran using PyTorch 1.7.0 , CUDA 10.1 and Python 3.6.9

## Usage

The modules [approx_Linear](src/pytorch/approx_mul_pytorch/modules/approx_Linear.py) and [approx_Conv2d](src/pytorch/approx_mul_pytorch/modules/approx_Conv2d.py) implement an approximate linear or conv2d layer, respectively. 

When initialized, the following parameters are used to control the amount of sampling:

```
sample_ratio - Ratio of column-row pairs to sample
minimal_k - Minimal number of column-row pairs to keep in the sampling
```

The approximation algorithms are implemented in [approx_linear_forward_xA_b](src/pytorch/approx_mul_pytorch/functional/approx_linear.py) and [approx_conv2d_func_forward](src/pytorch/approx_mul_pytorch/functional/approx_conv2d.py)

In order to run with "forward sampling", i.e sampling in the forward pass and backpropagation through the sampled entries, make sure the ```forward``` function in the layer module has the following enabled and the other calls commented out:
```
return approx_conv2d_func_forward(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.sample_ratio, self.minimal_k)
```

"Forwads sampling" performed best in practice and was used to generate most of the results in the paper.

In order to run with "backward sampling", i.e doing a full forward pass and approximating in the backward pass only, make sure the ```forward``` function in the layer module has has the following enabled and the other calls commented out:
```
return approx_conv2d_func.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.sample_ratio, self.minimal_k, self.sample_ratio, self.minimal_k, self.sample_ratio, self.minimal_k)
```

For "black box" forward sampling, i.e sampling in the forward pass but propagating gradients for all entries, use the following call in the module ```forward``` function:
```
return approx_conv2d_func.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.sample_ratio, self.minimal_k, None, None, None, None)
```

Note that for the convolution backward functions, the [custom cpp extensions](src/pytorch/cpp) need to first be installed using ```python setup.py install```.


Similar options can be used for the approximate linear layer.

The module [approx_Conv2d_dist](src/pytorch/approx_mul_pytorch/modules/approx_Conv2d_dist.py) samples according to the weight norms only. It is used by the multinode experiments in [imagenet_dist](src/pytorch/imagenet_dist) which implement a custom AllReduce scheme that performs reduction only on the sampled gradients. This allows to reduce the gradient communicataion traffic in multi-node training. 


## Results

We apply approximate tensor operations to single and multi-node training of MLP and CNN networks on MNIST, CIFAR-10 and ImageNet datasets. We demonstrate up to 66\% reduction in the amount of computations and communication, and up to 1.37x faster training time while maintaining negligible or no impact on the final test accuracy. Further details are described in the paper.

![fig_learning_curves](https://user-images.githubusercontent.com/18640225/138027903-0d56f491-7f97-4a22-800c-91eeb504dcf9.png)

