import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 训练集
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)

# 将训练集数据可视化
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
plt.close()


# 定义神经网络类
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):  #定义神经网络的结构
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  #定义hidden层功能
        self.predict = torch.nn.Linear(n_hidden, n_output)  #定义预测层功能

    def forward(self, x):                               #定义向前传播算法
        x = F.relu(self.hidden(x))  #使用hidden层
        x = self.predict(x)         #调用预测层
        return x


net = Net(1, 10, 1)
# print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 优化器，用于优化神经网络的参数，学习效率0.5
loss_func = torch.nn.MSELoss()  # 使用均值方差计算 损失函数

# 开始训练
for t in range(200):  # 训练200次
    prediction = net(x)  # 使用神经网络的参数计算预测值
    loss = loss_func(prediction, y)  # 使用预测值计算损失
    optimizer.zero_grad()  # 将梯度重置
    loss.backward()  # 使用反向传播计算节点梯度
    optimizer.step()  # 用优化器优化梯度
    # 可视化
    if t % 20 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)
        plt.pause(0.5)
        plt.close()