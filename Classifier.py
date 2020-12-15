# 二分类问题

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 准备训练数据
n_data = torch.ones(100,2)              # 生成100*2,值为1的矩阵
x0 = torch.normal(2*n_data,1)           # 生成正态分布x0,100个，均值为2，标准差为1
y0 = torch.zeros(100)                   # 数据x0的标签都为0
x1 = torch.normal(-2*n_data,1)          # 生成正太分布的x1,100个，均值为-2，标准差为1
y1 = torch.ones((100))                  # 数据x1的标签都为1
x = torch.cat((x0,x1),0).type(torch.FloatTensor)  # 将x0,x1合并，并将类型改为torch.FloatTensor
y = torch.cat((y0,y1)).type(torch.LongTensor)     # 将y0,y1合并，并将类型改为torch.LongTensor

x,y = Variable(x),Variable(y)           # x,y放到神经网络中学习

# 训练数据可视化
plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
plt.show()

# 定义神经网络 method1
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)       # 定义 hidden层 的数据处理方式
        self.predict = torch.nn.Linear(n_hidden,n_output)       # 定义 predict层 的数据处理方式

    def forward(self,x):
        x = F.relu(self.hidden(x))      # 使用 hidden层处理数据
        x = self.predict(x)             # 使用 predict层处理数据
        return x                        # x作为Net的预测结果输出


# 定义神经网络 method2
# net2 = torch.nn.Sequential(
#     torch.nn.Linear(2,10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10,2)
# )
# # 定义神经网络
# net2 = Net(2, 10, 2)
# optimizer = torch.optim.SGD(net2.parameters(),lr=0.01)       # 定义优化器
# loss_func = torch.nn.CrossEntropyLoss()                      # Classify问题使用的损失函数


# 定义神经网络
net = Net(2, 10, 2)
optimizer = torch.optim.SGD(net.parameters(),lr=0.01)       # 定义优化器
loss_func = torch.nn.CrossEntropyLoss()                       # Classify问题使用的损失函数

for t in range(50):
    prediction = net2(x)                     # 计算预测值
    loss = loss_func(prediction, y)         # 计算损失
    optimizer.zero_grad()                   # 梯度置零
    loss.backward()                         # 计算梯度
    optimizer.step()                        # 优化网络
    if t % 5 == 0:
        plt.cla()                           # Clear the current axes.
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(prediction), 1)[1]     # 将F.softmax()将prediction转换为概率eg:[0.2,0.8]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.               # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)
        plt.close()


    plt.ioff()                              # 停止画图

