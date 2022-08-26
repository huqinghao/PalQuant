# PalQuant
This is the offical implementation of  ["PalQuant: Accelerating High-precision Networks on Low-precision Accelerators" in ECCV 2022](https://arxiv.org/abs/2208.01944)



### Installation
```
git clone git@github.com:huqinghao/PalQuant.git
cd PalQuant
pip install -r requirements.txt
```

### Training
To train resnet-18 with 4-bit weights and activations, PalQuant trains a wide 2-bit resnet-18 with group=2:
```
python main.py --data your-imagenet-data-path --visible_gpus '0,1,2,3' --workers 20  \
--arch 'resnet18_quant' --epochs 90 --groups 2 --weight_levels 4 --lr_m 0.1 --lr_q 0.0001 \
-b 256  --act_levels 4 --log_dir "../results/resnet-18/W2A2G2/"
```
Here weight_levels and activations  levels equals $2^{bit}$


### Testing
We have uploaded the training checkpoint to the BaiduNetdisk and Google Storge. To test the pre-trained model, run:
```
python main.py --data your-imagenet-data-path --visible_gpus '0' --workers 20 \
--arch 'resnet18_quant'  --groups 2  --weight_levels 4 -b 256  --act_levels 4 \
--evaluate  --model the-model-to-eval
```

|  Model  | Weight Bits  | Act Bits | Groups | DownloadUrl | TensorboardLog |
|  ----  | ----  |----  |----  |----  |----  |
|  ResNet18  |  2 | 2 | 2 |  [BaiduNetDisk](https://pan.baidu.com/s/1mP7MqmiDGFdwekQjv9Lgk) |[log](/logs/W2A2G2/events.out.tfevents.1646237746.officer-AS-4124GS-TNR.507209.0)|
|  ResNet18  |  2 | 2 | 3 |  [BaiduNetDisk](https://pan.baidu.com/s/1cwV5f5nGKKUO6EJhV1Hk3A) |[log](/logs/W2A2G3/events.out.tfevents.1646399329.officer-AS-4124GS-TNR.3888486.0)|
|  ResNet18  |  2 | 2 | 4 |  [BaiduNetDisk](https://pan.baidu.com/s/1LLaXl_K8zLL1yeiZuK-BzQ) |[log](/logs/W2A2G4/events.out.tfevents.1646484476.gpu03.467963.0)|
extract code: quan


### Citation
```
@inproceedings{PalQuant2021,
  author  = {Qinghao Hu and Gang Li and Qiman Wu and Jian Cheng},
  title   = {PalQuant: Accelerating High-precision Networks on Low-precision Accelerators},
  year    = {2022},
  booktile={European Conference on Computer Vision},
  organization={Springer}
}
```
### Acknowledgement

The code base is origined from [EWGS](https://github.com/cvlab-yonsei/EWGS), we thank their awesome work.
