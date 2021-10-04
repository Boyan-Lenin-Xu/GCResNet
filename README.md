# GCResNet
PyTorch implementation of the paper "Graph Convolutional Networks in Feature Space for Image Deblurring and Super-resolution", IJCNN 2021.


## Code

Clone this repository and enter the file

    git clone https://github.com/Boyan-Lenin-Xu/GCResNet
    cd GCResNet

## How to train GCResNet (Deblurring)
    
    demo.sh

Use the last line:

    python ./src/main.py --mode deblur --save_results


## How to train GCEDSR (Super-resolution)

    demo.sh

Use:

    python ./src/main.py --mode super-resolution --model GCSR --scale 2 --save gcsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --save_results

## Test image deblurring

    ./options.xml

Set <mode> to Test, set <test_only> to True

## Paper and Cite

Xu B, Yin H. Graph Convolutional Networks in Feature Space for Image Deblurring and Super-resolution[C]//International Joint Conference on Neural Networks 2021. IEEE, 2021.

    @inproceedings{xu2021graph,
      title={Graph Convolutional Networks in Feature Space for Image Deblurring and Super-resolution},
      author={Xu, Boyan and Yin, Hujun},
      booktitle={International Joint Conference on Neural Networks 2021},
      year={2021},
      organization={IEEE}
    }
