# Sparsity-Regularized Out-of-distribution Detection

This repository is the implementation of [Improving Energy-based OOD Detection by Sparsity Regularization](https://link.springer.com/chapter/10.1007/978-3-031-05936-0_42) by Qichao Chen, Wenjie Jiang, Kuan Li and Yi Wang. This method is a simple yet effective for improve Energy-based OOD Detection. Our code is modified from [energy_ood](https://github.com/wetliu/energy_ood).

![image](https://github.com/ChurchChen/SparsityRegularization/blob/main/demo_fig/framework.pdf)

## Requirements

It is tested under Ubuntu Linux 18.04 and Python 3.7 environment, and requries some packages to be installed:

- PyTorch 1.4.0
- torchvision 0.5.0
- numpy 1.17.2

## Training Pretrained Models

Please download the datasets in folder

```shell
./data/
```

Training pretrained classifier

```shell
python baseline.py cifar10
python baseline.py cifar100
```

Pretrained models are provided in folder

```shell
./CIFAR/snapshots/
```

## Testing and Fine-tuning

Evaluate the pretrained model using energy-based detector

```shell
python test.py --model cifar10_wrn_pretrained --score energy
python test.py --model cifar100_wrn_pretrained --score energy
```

Fine-tune the pretrained model

```shell
python tune.py cifar10 --save ./snapshots/tune_sr
python tune.py cifar100 --save ./snapshots/tune_sr
```

Testing the detection performance of fine-tuned model 

```shell
python test.py --model cifar10_wrn_s1_tune --score energy
python test.py --model cifar100_wrn_s1_tune --score energy
```



## Results

Our model achieves the following average performance on 6 OOD datasets:

### 1. MSP vs energy score with and without fine-tuned on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

| Model name         |     FPR95       |
| ------------------ |---------------- |
| [MSP](https://arxiv.org/abs/1610.02136) |     51.35%     |
| [ODIN](https://arxiv.org/abs/1706.02690) |     35.59%     |
| [Mahalanobis](https://arxiv.org/abs/1807.03888) |     37.08%     |
| [EBD](https://arxiv.org/abs/2010.03759) |     33.01%     |
| SR (Ours) | 19.19% |

### 2. CIFAR-10 (in-distribution) vs SVHN (out-of-distribution) Score Distributions

![image](https://github.com/ChurchChen/SparsityRegularization/blob/main/demo_fig/energy_score_density.pdf)

### 3. Performance among different baselines for [WideResNet](https://arxiv.org/abs/1605.07146)
CIFAR-10:
| Method    |     FPR95       |
| ------------------ |---------------- |
| [Baseline](https://arxiv.org/abs/2010.03759) |     34.92%     |
| [Outlier Exposure](https://arxiv.org/abs/1812.04606) |     8.53%     |
| [Energy](https://arxiv.org/abs/2010.03759) |     3.32%     |
| SROE (Ours) | 4.15% |

CIFAR-100:

| Method    |     FPR95       |
| ------------------ |---------------- |
| [Baseline](https://arxiv.org/abs/2010.03759) |     71.86%     |
| [Outlier Exposure](https://arxiv.org/abs/1812.04606) |     56.57%     |
| [Energy](https://arxiv.org/abs/2010.03759) |     49.28%     |
| SROE (Ours) | 23.84% |

![image](https://github.com/ChurchChen/SparsityRegularization/blob/main/demo_fig/acc_auc.png)



## Outlier Datasets

These experiments make use of numerous outlier datasets. Links for less common datasets are as follows, [80 Million Tiny Images](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin) [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Places365](http://places2.csail.mit.edu/download.html), [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz), [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz) and [SVHN](http://ufldl.stanford.edu/housenumbers/).

Our **tiny** dataset available at [here](https://drive.google.com/file/d/1zKzzTkbJjODC_y5ZSY8RQAGzzEGqZhuj/view?usp=sharing)

![image](https://github.com/ChurchChen/SparsityRegularization/blob/main/demo_fig/tiny.pdf)

## Citation

     @article{chen2022sparsity,
          title={Improving Energy-based Out-of-distribution Detection by Sparsity Regularization},
          author={Chen, Qichao and Jiang, Wenjie and Li, Kuan and Wang, Yi},
          journal={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
          year={2022}
     } 
