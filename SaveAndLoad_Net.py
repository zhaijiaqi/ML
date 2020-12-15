import torch
from torch.autograd import Variable
import torch.functional as F
import matplotlib.pyplot as plt

# 生成假数据
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())

# 定义网络并保存
def define_save_net():
    # 定义神经网络
    net1 = torch.nn.Sequential(         # 快速搭建神经网络
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net1.parameters(),lr=0.2)   # 定义神经网络的优化器
    loss_func = torch.nn.MSELoss()                          # 定义神经网络的损失函数
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # plot result
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

    #保存神经网络到两个文件中
    torch.save(net1,'./pkl/net.pkl')
    torch.save(net1.state_dict(),'./pkl/net_params.pkl')


def restore_net():
    net2 = torch.load('./pkl/net.pkl')
    prediction = net2(x)
    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('./pkl/net_params.pkl'))
    prediction = net3(x)
    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

define_save_net()
restore_net()
restore_params()

