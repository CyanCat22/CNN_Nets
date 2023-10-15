import torch
from torch import nn 
from d2l import torch as d2l

"""
批量归一化
固定小批量里面的均值和方差
调整额外的可学习的参数,学习出适合的偏移和缩放
可以加速收敛速度,但一般不改变模型精度
"""

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # eps:很小的数,防止除以0的情况发生 momentum:动量参数,控制移动平均的更新速度
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 预测模式
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 训练模式
        assert len(X.shape) in (2, 4)
        # 这段代码是一个断言语句，用于检查变量X的形状（shape）是否为2维或4维。如果X的形状不是2维或4维，程序将抛出异常并终止执行。
        if len(X.shape) == 2:
            # 全连接层 1维是通道,2维是特征
            mean = X.mean(dim = 0) # 按行求均值
            var = ((X-mean) ** 2).mean(dim = 0) # 求方差
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim = True)
            var = ((X.mean) ** 2).mean(dim=(0, 2, 3), keepdim = True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)            
        # 更新均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum)*mean
        moving_var = momentum * moving_var + (1.0 - momentum)*var
        # 当momentum接近1时，移动平均会快速地收敛到历史数据的真实均值和方差；
        # 当momentum接近0时，移动平均会较慢地收敛到历史数据的真实均值和方差。
        # 通过调整momentum的值，我们可以在计算效率和模型性能之间找到一个平衡点
        
    Y = gamma * X_hat + beta # 缩放和移动
    return Y, moving_mean.data, moving_var.data
        
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        # 将移动平均均值初始化为0，表示初始时我们认为数据的均值为0；
        # 将移动平均方差初始化为1，表示初始时我们认为数据是均匀分布的，方差为1。
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps = 1e-5, momentum=0.9)
        return Y
        
# LeNet应用BatchNorm
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

# 学习率大很多
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

