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

In order to run with sampling in the forward pass and backpropagation through the samplnig 


To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
