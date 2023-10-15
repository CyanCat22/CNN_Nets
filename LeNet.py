import torch
from torch import nn 
from d2l import torch as d2l
"""
卷积神经网络
使用卷积层来学习图片空间信息
使用全连接层来转换到类别空间
"""
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    
net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(), # 变成一个一维向量
    nn.Linear(16*5*5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# GPU计算评估
def evaluate_acc_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        # 检查net是否为nn.Module的实例
        net.eval() # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
            # 如果没有提供device参数，那么将使用模型中第一个参数所在的设备作为评估设备

    metric = d2l.Accumulator(2) #累加器，累积正确预测的数量和总预测的数量

    with torch.no_grad(): # 上下文管理器 确保在评估过程中不计算梯度
        for X, y in data_iter:
            if isinstance(X, list): 
                # 如果X是一个列表，说明输入数据是批量的，需要对每个样本进行单独处理。将列表中的每个元素转换为设备（GPU）上的张量
                X = [x.to(device)for x in X]
            else:
                X = X.to(device) # 转换为设备(GPU)上的张量
            y = y.to(device)
            
            metric.add(d2l.accuracy(net(X), y), y.numel()) 
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d :
            nn.init.xavier_uniform_(m.weight)
            # Xavier 初始化的核心思想是保持每一层输出的方差与输入的方差一致，以防止信号在深层网络中的爆炸或消失
    net.apply(init_weights) # 将初始化函数应用到网络的每一个模块
    print("train on", device) # 打印一下是否在GPU上训练
    net.to(device) # 将网络设置为GPU
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 损失函数为交叉熵损失
    loss = nn.CrossEntropyLoss()
    # 创建动画展示器，用于绘制训练过程中的损失、准确率变化曲线
    animator = d2l.Animator(xlabel='epoch', xlim = [1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    # 计时                            
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train() # 将网络设置为训练模式
        for i, (X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X) # 计算网络输出
            l = loss(y_hat, y)      
            l.backward() # 反向传播，计算梯度
            optimizer.step() # 更新参数
            # 在不计算梯度的情况下，计算准确率并添加到累积器中
            with torch.no_grad():
                metric.add(l*X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            # 计算平均训练损失和准确率
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # 每隔5个批次或到达最后一个批次时，更新动画展示器
            if(i+1)%(num_batches//5) == 0 or i == num_batches - 1:
                animator.add(epoch+(i+1)/num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_acc_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '    
            f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}')

lr, num_epochs = 0.9, 10
train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# 运行成功，但是没有显示animator