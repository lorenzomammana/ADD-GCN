# ADD-GCN: Attention-Driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition

Reimplementation of the ADD-GCN code available [here](https://github.com/Yejin0111/ADD-GCN) with updated Pytorch 
version, support for hydra configuration and Pytorch lightning.
Now the code supports changing backbones, and returns the dynamic adjacency matrix for a given input image.
Also a flag has been included to replace ADD-GCN with a standard MLP for comparison.

    Attention-Driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition;
    Jin Ye, Junjun He, Xiaojiang Peng, Wenhao Wu, Yu Qiao;
    In: European Conference on Computer Vision (ECCV), 2020.
    arXiv preprint arXiv:2012.02994 

The full paper is available at: [https://arxiv.org/abs/2012.02994](https://arxiv.org/abs/2012.02994). 

## Installation
#### This project is implemented with Pytorch Lightning and has been tested on version Pytorch 1.10 - Lightning 1.5.5.

## A quick demo
After you have installed the requirements, you can follow the below steps to run a quick demo.

### Train for COCO2014
    python run_train.py
    
Hydra configuration is provided into configs/train.yaml

### Evaluation for COCO2014


Please note that:
1) You should put the COCO2014 folder in {YOUR-ROOT-DATA-DIR}.

2) You should put the test model in {THE-TEST-MODEL} folder.

3) You can get the same ADD-GCN results with [this model](https://pan.baidu.com/s/17Y1knACAo5U6XbV75GUI8w). The password is ``4ebj``.

Model | Test size | mAP 
--- |:---:|:---:
ResNet-101 | 448×448 | 79.7
DecoupleNet | 448×448 | 82.2
ML-GCN | 448×448 | 83.0 
ADD-GCN | 448×448 | 84.2
ResNet-101 | 576×576 | 80.0
SSGRL | 576×576 | 84.2
ML-GCN | 576×576 | 84.3
ADD-GCN | 576×576 | 85.2


## Citations
Please consider citing authors paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{ye2020add,
  title   =  {Attention-Driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition},
  author  =  {Jin Ye, Junjun He, Xiaojiang Peng, Wenhao Wu, Yu Qiao},
  booktitle =  {European Conference on Computer Vision (ECCV)},
  year    =  {2020}
}
```


## License

