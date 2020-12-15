import torch
import torch.nn.functional as F
import torch.utils.data as D
from torch.autograd import Variable
import matplotlib.pyplot as plt

def main():
    # hyper parameters
    LR = 0.1
    EPOCH = 12
    BATCH_SIZE = 32

    # data
    x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    y = x.pow(2)+0.1*torch.rand(x.size())

    # plot data
    plt.scatter(x.numpy(),y.numpy())
    plt.show()

    # 批处理需要loader
    torch_dataset = D.TensorDataset(x, y)
    loader = D.DataLoader(
        dataset = torch_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2,
    )

    # default net
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            self.hidden = torch.nn.Linear(1,10)
            self.predict = torch.nn.Linear(10,1)
        def forward(self,x):
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x

    # 4 different net
    net_SGD =        Net()
    net_Momentum =   Net()
    net_RMSprop =    Net()
    net_Adam =       Net()
    nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]

    # 4 different optimizer
    opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()      # define loss function
    losses_his = [[],[],[],[]]          # recode the history of the loss in every net

    for epoch in range(EPOCH):
        print('EPOCH:', epoch)
        for step, (batch_x,batch_y) in enumerate(loader):
            batch_x = Variable(batch_x)
            batch_y = Variable(batch_y)
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                out = net(batch_x)
                loss = loss_func(out,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())  # loss recoder

    labels = ['SGD','Momentum','RMSprop','Adam']
    for i,l_his in enumerate(losses_his):
        plt.plot(l_his,label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.5))
    plt.show()


if __name__ == '__main__':
    main()