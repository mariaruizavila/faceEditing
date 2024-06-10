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
Download the training data and place it under a directory `prepared_data/train/`. Download as well the testing data and place ir under a directory `data/test/`.

You then need to map the test dataset to latent codes (the linked training data is already mapped). To embed the images into the latent space of StyleGAN2, we have used the Image2StyleGAN encoder. Download it from [this repository](https://github.com/eladrich/pixel2style2pixel) and download as well the [pretrained latent classifier](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view)). Save the pretrained model in `pretrained_models/`. Run the following command:

```sh
cd pixel2style2pixel/
python scripts/inference.py \
--checkpoint_path=pretrained_models/psp_ffhq_encode.pt \
--data_path=../data/test/ \
--exp_dir=../prepared_data/test/ \
--test_batch_size=1
```

When using the UI, it is also required to map the input image to its latent code. All the images uploaded by the user are stored in `data/input/` and they are internally prepared afterwards.

### Train all the models
For doing that download the code from the linked github repositories, save them under the directory `models/` and follow the explained instructions in their README to train them. Make sure you train them with the prepared training dataset. Note that it is not neccessary to train the text encoders in StyleCLIP and TediGAN, you can take a pretrained CLIP instead to replace the visual-linguistic learning module.

### Calculate the metrics

+ File `evaluation/random.py` contains a function to calculate a random set of attributes to be changed for each sample of the testing dataset.
+ File `evaluation/statistics.py` contains the functions we have used in order to calculate the Change Ratio, Identity Preservation and Attribute Preservation. This metrics are proposed in [this work](https://github.com/adriacarrasquilla/latent-multi-transformer?tab=readme-ov-file).
+ File `evaluation/performance.py` contains the function we have used to evaluate the performance of the models.
+ File `evaluation/attribute_correlation.ipynb` contains the attibute correlation analysis we have made.
+ File `evaluation/probability.ipynb` contains the graphs to visualize the changes of the probability of a picture having a certain attribute before and after changing non-targeted attributes.

In order to get the desired metrics you just need to run the following commnads:

```sh
cd evaluation/
python statistics.py \
-- model_path=../models/path/to/model
```

```sh
cd evaluation/
python performance.py \
-- model_path=../models/path/to/model
```

You can also run the jupyter notebooks to visualize the data.

### Run the UI

```sh
python UI/initial.py
```

## Structure

```plaintext
├── data
│   ├── test
│   ├── input
│   └── output
├── prepared_data
│   ├── train
│   ├── test
│   ├── input
│   └── output
├── models
│   ├── InterFaceGAN
│   ├── TediGAN
│   ├── StyleCLIP
│   ├── pixel2style2pixel
│   └── latent-multi-transformer
├── evaluation
│   ├── random.py
│   ├── statistics.py
│   ├── performance.py
│   ├── attribute_correlation.ipynb
│   └── probability.ipynb
└── UI
    ├── main.py
    └── initial.py
```
