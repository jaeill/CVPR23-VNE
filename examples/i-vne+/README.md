



# I-VNE+: A novel method for unsupervised representation learning

## I-VNE+ in ImageNet-100


#### Training

```bash
python -W ignore -u train_ivne_imagenet100.py --gpu_num 0,1 --datadir ./data/imagenet --cache_name I_VNE_ImageNet_100
```

#### Evaluation

```bash
python -W ignore -u eval_ivne_imagenet100.py --gpu_num 0 --datadir ./data/imagenet --cache_name I_VNE_ImageNet_100
```





