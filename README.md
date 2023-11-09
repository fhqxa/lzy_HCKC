## Hierarchical convolutional neural network with knowledge complementation for long-tailed classification
HONG ZHAO*, ZHENGYU LI, WENWEI HE, and Yan Zhao 

## Main requirements

  * **torch == 1.0.1**
  * **torchvision == 0.2.2_post3**
  * **tensorboardX == 1.8**
  * **Python 3**


## Environmental settings
The code is developed using the PyTorch framework. We conduct experiments on a single NVIDIA GeForce RTX 2080 Ti GPU. 
The CUDA nad CUDNN version is **9.0** and **7.1.3** respectively.
Other platforms or GPU cards are not fully tested.


## Usage
```bash
# To train long-tailed CIFAR-10 with imbalanced ratio of 50:
python main/train.py  --cfg configs/cifar10.yaml     

# To validate with the best model:
python main/valid.py  --cfg configs/cifar10.yaml

# To debug with CPU mode:
python main/train.py  --cfg configs/cifar10.yaml   CPU_MODE True
```

You can change the experimental setting by simply modifying the parameter in the yaml file.

## Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `fpath`, `im_height`, `im_width` and `category_id`.


## Pretrain models for iNaturalist

We provide the BBN pretrain models of both 1x scheduler and 2x scheduler for iNaturalist 2018 and iNaturalist 2017.

iNaturalist 2018: [Baidu Cloud](https://pan.baidu.com/s/1olDppTptZ5HYWsgQsMCPLQ), [Google Drive](https://drive.google.com/open?id=1B9ZEfMHqE-KQRKX6nQLQRm8ErFrnHaoE)

iNaturalist 2017: [Baidu Cloud](https://pan.baidu.com/s/1soxsHKKblhapew_wuEdKPQ), [Google Drive](https://drive.google.com/open?id=1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU)

The experimental setup was as follows: 

````
python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4
````
[figure_3method1.pdf](https://github.com/fhqxa/lzy_HCKC/files/13304999/figure_3method1.pdf)

![image](https://github.com/fhqxa/lzy_HCKC/assets/36149734/3d486754-f1d0-4636-a948-0efd1a69e518)
