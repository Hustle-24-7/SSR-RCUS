# SSR-RCUS (AAAI2025) 

This is an official implementation of [Semi-supervised Regression by Preserving Ranking Relationships between Close Unlabeled Samples](https://openreview.net/pdf?id=4kIpkMV4Ho), which is accepted by AAAI 2025.

## Abstract

Semi-Supervised Learning (SSL) aims to improve the learning performance of supervised learning with a large number of unlabeled samples. The existing SSL methods such as FixMatch and FlexMatch select unlabeled samples with high-confident pseudo-labels and make consistency constraints between their weak and strong augmentations. Unfortunately, they cannot be applied Semi-Supervised Regression (SSR) because regression predictions can not reflect the confidence of pseudo-labels. To solve this, a recent SSR method RankUp incorporates an auxiliary ranking task by leveraging sample pairs with high-confident pseudo-ranks. In this paper, we upgrade RankUp to a novel SSR method, namely Semi-Supervised Regression by Ranking Close Unlabeled Samples (SSR-RCUS). Its basic idea is reconstructing closed mixup augmented samples with high-confident pseudo-ranks under a monotonicity assumption, and then applying them to the auxiliary ranking task to improve regression performance. We conduct extensive experiments to evaluate the performance of SSR-RCUS on benchmark datasets, and empirical results demonstrate that
SSR-RCUS can outperform the existing baselines in various settings, especially when labeled data are scarce.

# Training the Model 
You can train the model by using the config, for example, on utkface:

```
python train.py --gpu 0 --c config/classic_cv/rcus/rcus_utkface_lb10_s0.yaml
python train.py --gpu 0 --c config/classic_cv/rcus/rcus_utkface_lb50_s0.yaml
python train.py --gpu 0 --c config/classic_cv/rcus/rcus_utkface_lb250_s0.yaml
python train.py --gpu 0 --c config/classic_cv/rcus/rcus_utkface_lb2000_s0.yaml
```

