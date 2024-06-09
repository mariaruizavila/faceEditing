# Comparative Analysis of State-of-the-Art Deep-Learning Based Face Editing Algorithms
## Acknowledgements

Please note that not all code contained in this repository was written by me. This project incorporates work from other developers and sources. Wherever possible, I have endeavored to attribute the original authors and provide links to the source material.

## Dependencies

+ Python 3.10
+ Pytorch 1.1.0
+ CUDA
+ Gradio

## Dataset

### Training data

We have used the CelebA-HQ dataset to train the latent transformer. You can download the prepared dataset from [this drive]([URL](https://drive.google.com/drive/folders/1aXVc-q2ER7A9aACSwml5Wyw5ZgrgPq52)). It is extracted from the [A Latent Transformer for Disentangled Face Editing in Images and Videos](https://arxiv.org/pdf/2106.11895) paper. 

### Testing data

We have used the first 1K images from FFHQ to test the model and get the desired metrics. You can download them from [this drive](https://drive.google.com/drive/folders/1taHKxS66YKJNhdhiGcEdM6nnE5W9zBb1). 

## Compared Models

### InterFaceGAN

+ [Official Implementation](https://github.com/genforce/interfacegan)
+ [Paper](https://arxiv.org/pdf/2005.09635)

### TediGAN

+ [Official Implementation](https://github.com/IIGROUP/TediGAN)
+ [Paper](https://arxiv.org/pdf/2012.03308)

### StyleCLIP

+ [Official Implementation](https://github.com/orpatashnik/StyleCLIP)
+ [Paper](https://arxiv.org/abs/2103.17249)

### Single Latent Transformer

+ [Official Implementation](https://github.com/InterDigitalInc/latent-transformer)
+ [Paper](https://arxiv.org/pdf/2106.11895)

### Mutli-Attribute Latent Transformer

+ [Official Implementation](https://github.com/adriacarrasquilla/latent-multi-transformer?tab=readme-ov-file)

## Instructions

### Prepare the datasets
Download the training data and place it under a directory `data/train/`. Download as well the testing data and place ir under a directory `data/test/`.

You the need you need to map the datasets to latent codes. For doing this, we have used the Image2StyleGAN encoder to embed the images into the latent space. 

```sh
cd pixel2style2pixel/
python scripts/inference.py \
--checkpoint_path=pretrained_models/psp_ffhq_encode.pt \
--data_path=/path/to/ffhq/pngs/ \
--exp_dir=../data/test/ \
--test_batch_size=1
```

holi

### Prepare StyleGAN2 encoder and generator

### Train all the models
For doing that download the code from the linked github repositories and follow the explained instructions. 
