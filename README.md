### Skip-Attension-GANomaly-Pytorch
[(Back to top)](#table-of-contents)

Generator +  Discriminator model 


### Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [Skip-Attension-GANomaly-Pytorch](#Skip-Attension-GANomaly-Pytorch)
- [Requirement](#Requirement)
- [implement](#implement)
   - [CBAM-Convolutional-Block-Attention-Module](#CBAM-Convolutional-Block-Attention-Module)
      - [Channel-Attension-module](#Channel-Attension-module)
      - [Spatial-Attension-module](#Spatial-Attension-module)
- [Train-on-custom-dataset](#Train-on-custom-dataset)
- [Train](#Train)
- [Test](#Test)
- [Lose-value-distribution](#Lose-value-distribution)
- [Reference](#Reference)
   
### Requirement
```
pip install -r requirements.txt
```

### implement 
[(Back to top)](#table-of-contents)

- [Unet](#Unet)
- [CBAM-Convolutional-Block-Attention-Module](#CBAM-Convolutional-Block-Attention-Module)
   - [channel attension](#Channel-Attension-module)
   - [spatial attension](#Spatial-Attension-module)

### Unet
[(Back to top)](#implement)

![image](https://user-images.githubusercontent.com/58428559/196223327-51bacb0f-6490-491b-ab80-727329dcc84f.png)

### CBAM-Convolutional-Block-Attention-Module
[(Back to top)](#implement)

![image](https://user-images.githubusercontent.com/58428559/196224948-4d588ad7-f272-4e05-bc8e-9a205e9c69be.png)

#### Channel-Attension-module
[(Back to top)](#implement)

![image](https://user-images.githubusercontent.com/58428559/196225065-083d3863-ae64-47d1-b6db-b1618e947e03.png)

#### Spatial-Attension-module
[(Back to top)](#implement)

![image](https://user-images.githubusercontent.com/58428559/196225149-1ef408a6-18e8-4a8b-847d-a4471f6e6d2c.png)

### Train-on-custom-dataset
[(Back to top)](#table-of-contents)

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png


```

### Train
[(Back to top)](#table-of-contents)
```
python train.py --img-dir "[train dataset dir]" --batch-size 64 --img-size 32 --epoch 20
```
### Test
[(Back to top)](#table-of-contents)
```
python test.py --nomal-dir "[test normal dataset dir]" --abnormal-dir "[test abnormal dataset dir]" --view-img --img-size 32
```
Example :
Train dataset : factory line only

dataset :factory line , top: input images, bottom: reconstruct images
![infer_normal1](https://user-images.githubusercontent.com/58428559/196330429-57007e6e-cfb0-4159-b687-ec6c5f550bd9.jpg)


dataset :factory noline , top: input images, bottom: reconstruct images
![infer_abnormal3](https://user-images.githubusercontent.com/58428559/196330451-f29f997c-1481-4cbc-80d4-2a39d28afa06.jpg)


### Lose-value-distribution
[(Back to top)](#table-of-contents)

Blue : normal dataset

Orange : abnormal dataset

train 128x128

![loss_distribution](https://user-images.githubusercontent.com/58428559/196330466-ae2d021b-5401-4f9f-8a48-504b2aea9de1.jpg)



### Reference 
[(Back to top)](#table-of-contents)

#### GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training

https://arxiv.org/abs/1805.06725

#### CBAM: Convolutional Block Attention Module

https://arxiv.org/pdf/1807.06521.pdf

#### SAGAN: Skip-Attention GAN For Anomaly Detection

http://personal.ee.surrey.ac.uk/Personal/W.Wang/papers/LiuLZHW_ICIP_2021.pdf



