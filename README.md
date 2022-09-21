## Efficient Equivariant Network

* This repository provides official implementations for [Efficient Equivariant Network](https://proceedings.neurips.cc/paper/2021/file/2a79ea27c279e471f4d180b08d62b00a-Paper.pdf).
* If you have questions, please send an e-mail to us (lingshenhe@pku.edu.cn) or make an issue.

### Abstract
Convolutional neural networks (CNNs) have dominated the field of Computer Vi- sion and achieved great success due to their built-in translation equivariance. Group equivariant CNNs (G-CNNs) that incorporate more equivariance can significantly improve the performance of conventional CNNs. However, G-CNNs are faced with two major challenges: spatial-agnostic problem and expensive computational cost. In this work, we propose a general framework of previous equivariant models, which includes G-CNNs and equivariant self-attention layers as special cases. Un- der this framework, we explicitly decompose the feature aggregation operation into a kernel generator and an encoder, and decouple the spatial and extra geometric dimensions in the computation. Therefore, our filters are essentially dynamic rather than being spatial-agnostic. We further show that our Equivariant model is parameter Efficient and computational Efficient by complexity analysis, and also data Efficient by experiments, so we call our model E4-Net. Extensive experiments verify that our model can significantly improve previous works with smaller model size. Especially, under the setting of training on 1/5 data of CIFAR10, our model improves G-CNNs by 5%+ accuracy, while using only 56% parameters and 68% FLOPs.


### Requirement 
<pre>
python=3.8.5
tqdm
pytorch=1.8.1
torchvision=0.9.1
cuda=10.1
Device： 1 GTX1080Ti

</pre>
### Experiment Settings
#### MNIST-rot
* Architectures: Normal, Large
* Training batch size: 128
* Weight decay: 1e-4
* Input normalization
* Learning rate adjustment
  1) 0.02 for epoch [0, 60)
  2) 0.002 for epoch [60, 120)
  3) 0.0002 for epoch [120, 160)
  4) 0.00002 for epoch [160, 200)
 * Kernel size: 5
 * Reduction: 1 (for normal model), 2 (for large model)
 * groups (=channels/slices): 8 (for normal model), 2 (for large model)
 * dropout rate: 0.2

#### CIFAR10 and CIFAR100
* Architectures: resnet18, C4resnet18, E4C4resnet18, D4resnet18, E4D4resnet18
* Training batch size: 128
* Weight decay: 5e-4
* Input normalization
* Learning rate adjustment
  1) 0.1 for epoch [0, 60)
  2) 0.02 for epoch [60, 120)
  3) 0.004 for epoch [120, 160)
  4) 0.0008 for epoch [160, 200)
 * Kernel size: 3
 * Reduction: 2 (for normal model), 2 (for large model)
 * groups (=channels/slices): 2
 * dropout rate: 0.2

### Training

#### 1. MNIST-rot

<pre>

# E^4-Net(Normal)
CUDA_VISIBLE_DEVICES=0 python3 train_mnist.py --model normal --groups 8 --reduction 1

# E^4-Net(Large)

CUDA_VISIBLE_DEVICES=0 python3 train_mnist.py --model large --groups 2 --reduction 2
</pre>

||MNIST-rot|
|------|---
|Normal|98.71%
|Large|98.81%
#### 2,CIFAR

<pre>
# R18 on CIFAR
CUDA_VISIBLE_DEVICES=0  python3 train.py --model resnet18 --dataset cifar10
CUDA_VISIBLE_DEVICES=0  python3 train.py --model resnet18 --dataset cifar100

# p4-R18 on CIFAR
CUDA_VISIBLE_DEVICES=0 python3 train.py --model C4resnet18 --dataset cifar10
CUDA_VISIBLE_DEVICES=0 python3 train.py --model C4resnet18 --dataset cifar100

# p4-E^4R18 on CIFAR
CUDA_VISIBLE_DEVICES=0 python3 train.py --model E4C4resnet18 --dataset cifar10
CUDA_VISIBLE_DEVICES=0 python3 train.py --model E4C4resnet18 --dataset cifar100

# p4m-R18 on CIFAR
CUDA_VISIBLE_DEVICES=0 python3 train.py --model D4resnet18 --dataset cifar10
CUDA_VISIBLE_DEVICES=0 python3 train.py --model D4resnet18 --dataset cifar100

# p4m-E^4R18 on CIFAR
CUDA_VISIBLE_DEVICES=0 python3 train.py --model E4D4resnet18 --dataset cifar10
CUDA_VISIBLE_DEVICES=0 python3 train.py --model E4D4resnet18 --dataset cifar100


</pre>
||CIFAR-10|CIFAR-100|
|------|---|---|
|R18|90.3%|66%|
|p4-R18|92.47%|72.04%|
|p4-E4R18|**93.58%**|**73.41%**|
|p4m-R18|94.17%|75.05%|
|p4m-E4R18|**95.04%**|**77.82%**|

## Citations

```bibtex
@inproceedings{
he2021efficient,
title={Efficient Equivariant Network},
author={Lingshen He and Yuxuan Chen and Zhengyang Shen and Yiming Dong and Yisen Wang and Zhouchen Lin},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=4-Py8BiJwHI}
}
```








