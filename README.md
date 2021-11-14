# GCResNet
PyTorch implementation of the paper [Graph Convolutional Networks in Feature Space for Image Deblurring and Super-resolution](http://arxiv.org/pdf/2105.10465.pdf), IJCNN 2021.

Boyan Xu, Hujun Yin

## Overview

Graph convolutional networks (GCNs) have achieved great success in dealing with data of non-Euclidean structures. Their success directly attributes to  fitting graph structures effectively to data such as in social media and knowledge databases. For image processing applications, the use of graph structures and GCNs have not been fully explored. In this paper, we propose a novel encoder-decoder network with added graph convolutions by converting feature maps to vertexes of a pre-generated graph to synthetically construct graph-structured data. By doing this, we inexplicitly apply graph Laplacian regularization to the feature maps, making them more structured. The experiments show that it significantly boosts performance for image restoration tasks, including deblurring and super-resolution. We believe it opens up opportunities for GCN-based approaches in more applications. 


## Datasets

GoPro
DIV2K
Set5
Set14
BSD100
Urban100


## Code

Clone this repository and enter the file

    git clone https://github.com/Boyan-Lenin-Xu/GCResNet
    cd GCResNet

Datasets should be put in ./dataset/

## How to train GCResNet (Deblurring)

Most of the options for image deblurring are in options.xml. You can modify options.xml to change the settings.
Use ./demo.sh to select the mode. Then use the following code to train GCResNet.
    
    demo.sh

Or use:

    python ./src/main.py --mode deblur --save_results


## How to train GCEDSR (Super-resolution)

Most of the options for image super-resolution are in ./src/option.py. You can modify ./src/option.py to change the settings. Then use the following code to train GCEDSR.

    demo.sh

Or use:

    python ./src/main.py --mode super-resolution --model GCSR --save_results

You can also use the code to train EDSR.

## Test

To test image deblurring, use ./options.xml to set the options.

    ./options.xml

    Set <mode> to Test, set <test_only> to True

To test image super-resolution, use demo.sh or src/option.py to set --test_only.

## Paper and Cite

Xu B, Yin H. Graph Convolutional Networks in Feature Space for Image Deblurring and Super-resolution[C]//International Joint Conference on Neural Networks 2021. IEEE, 2021.

    @inproceedings{xu2021graph,
      title={Graph Convolutional Networks in Feature Space for Image Deblurring and Super-resolution},
      author={Xu, Boyan and Yin, Hujun},
      booktitle={International Joint Conference on Neural Networks 2021},
      year={2021},
      organization={IEEE}
    }
