# jittor-TensoRF

## 简介
TensoRF的jittor实现，本项目用于第二届计图人工智能挑战赛

## 安装

#### Jittor安装
根据运行环境的配置，按照jittor官网的[教程](https://cg.cs.tsinghua.edu.cn/jittor/)安装jittor并进行测试

#### 安装依赖
```
pip install tqdm scikit-image opencv-python configargparse imageio-ffmpeg tensorboardX
```

## 运行
#### 训练
```
python train.py --config configs/Scar.txt
```
#### 提取三维模型
```
python train.py --config configs/Scar.txt --ckpt path/to/your/checkpoint --export_mesh 1
```
## TODO:
1.速度与pytorch版本相比较慢，排查这一原因

2.目前只能处理blender数据集，其他数据集的相关代码还未修改