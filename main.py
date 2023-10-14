import torch 
from torch import nn 
from d2l import torch as d2l

def corr2d(X, K):
    """计算二维相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) # 输出结果
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y

class Conv2D(nn.Module):
    """实现二维卷积层"""
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

def forward(self, x):
    return corr2d(x, self.weight) + self.bias

"""Padding Stride 填充和步幅"""
"""多个输入通道"""
"""Pooling 
最大池化层---输出窗口内最大值
平均池化层"""
def pool2d(X, pool_size, mode = 'max'):
    




