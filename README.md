# PalQuant
This is the offical implementation of  ["PalQuant: Accelerating High-precision Networks on Low-precision Accelerators" in ECCV 2022]()



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
We have uploaded the training checkpoint to the Baidu Cloud and Google Storge. To test the resnet-18 model, download model from [BaiduNetdisk](https://pan.baidu.com/s/1SCk8xA1SVe5UwJ_l4ReZFw)(extract code: quan), and run:
```
python main.py --data your-imagenet-data-path --visible_gpus '0' --workers 20 \
--arch 'resnet18_quant'  --groups 2  --weight_levels 4 -b 256  --act_levels 4 \
--evaluate  --model the-model-to-eval
```

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
