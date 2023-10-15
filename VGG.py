import torch
from torch import nn 
from d2l import torch as d2l

"""
VGG_bolck
带填充以保持分辨率的卷积层
非线性激活函数,如ReLU
汇聚层，如最大汇聚层。
更大更深的AlexNet
"""
def vgg_block(num_convs, in_channels, out_channels):
    layers = [] # 存储卷积层和激活函数
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 最大池化层，减小特征图的空间尺寸
    return nn.Sequential(*layers) # 将layers列表中的所有层组合成一个序列

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for(num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels # 更新
    return nn.Sequential(
        *conv_blks, nn.Flatten(), # 将特征图展平为一维向量
        nn.Linear(out_channels*7*7, 4096), nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
    
net = vgg(conv_arch)
# X = torch.randn(size = (1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__,'output shape:\t',X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
