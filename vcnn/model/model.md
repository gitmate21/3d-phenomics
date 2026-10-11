# Pretrained model

The pretrained model used in the original study was distributed by the authors via
Baidu Pan (link in the original repository). Those links are often dead.

With this reproduction you can simply **train your own** model:

```bash
python scripts/train.py --data data/x_y1_y2_idx_33_withoutnoise.h5 --model vcnn_demo --epochs 5
```

Checkpoints are saved as `<model>_<epoch>.pth` and `<model>_final.pth` in the
checkpoint directory (see `config.py`). Use `--resume <epoch>` to continue training.

If you use this work, please cite:

Jin S, Su Y, Gao S, Wu F, Ma Q, Xu K, Ma Q, Hu T, Liu J, Pang S, Guan H, Zhang J,
Guo Q. 2019. Separating the structural components of maize for field phenotyping
using terrestrial lidar data and deep convolutional neural networks.
IEEE Transactions on Geoscience and Remote Sensing.
