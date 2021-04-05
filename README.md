# Revisiting Dynamic Convolution via Matrix Decomposition (ICLR 2021)
A [pytorch](http://pytorch.org/) implementation of [DCD](https://arxiv.org/abs/2103.08756).
If you use this code in your research please consider citing
>@article{li2021revisiting,
  title={Revisiting Dynamic Convolution via Matrix Decomposition},
  author={Li, Yunsheng and Chen, Yinpeng and Dai, Xiyang and Liu, Mengchen and Chen, Dongdong and Yu, Ye and Yuan, Lu and Liu, Zicheng and Chen, Mei and Vasconcelos, Nuno},
  journal={arXiv preprint arXiv:2103.08756},
  year={2021}
}
### Requirements

- Hardware: PC with NVIDIA Titan GPU.
- Software: *Ubuntu 16.04*, *CUDA 10.0*, *Anaconda2*, *pytorch 1.0.0*
- Python package
  - `conda install --quiet --yes pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch`
  - `pip install tensorboard tensorboardX pillow==6.1`


### Evaluate DCD on ImageNet

The pre-trained model can be downloaded here [ResNet-50](https://drive.google.com/file/d/14VUXecopj3aTu1s4IKdT2FsPt7Iq7BDK/view?usp=sharing) and [MobileNetV2x1.0](https://drive.google.com/file/d/1Nc8VsUTpm8NwWthwJUD75deJCbE_LHvk/view?usp=sharing)

DCD for ResNet-50

```
python main.py -a resnet50_dcd -d /path/to/imagenet/ -b 256 -c /path/to/output -j 48 --input-size 224 --dropout 0.1 --weight /path/to/resnet50_dcd.pth.tar --evaluate

```

DCD for MobileNetV2x1.0

```
python main.py -a mobilenetv2_dcd -d /path/to/imagenet/ -b 512 -c /path/to/output --width-mult 1.0 -j 48 --input-size 224 --dropout 0.1 --fc-squeeze 16 --weight mv2x1.0_dcd.pth.tar --evaluate

```

### Train DCD on ImageNet

DCD for ResNet-50

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50_dcd -d /path/to/imagenet/ -b 256 --epochs 120 --lr-decay schedule --lr 0.1 --wd 1e-4 -c /path/to/output -j 48 --input-size 224 --label-smoothing 0.1 --dropout 0.1 --mixup 0.2
```

DCD for MobileNetV2x1.0
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -a mobilenetv2_dcd -d /path/to/imagenet/ --epochs 300 --lr-decay cos --lr 0.1 --wd 2e-5 -c /path/to/output --width-mult 1.0 -j 48 --input-size 224 --label-smoothing 0.1 --dropout 0.2 -b 512 --mixup 0.2 --fc-squeeze 16
```
