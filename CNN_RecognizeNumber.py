import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data

# hyper parameters
EPOCH = 1                       # 批训练周期数
LR = 0.001                      # 学习速率
BATCH_SIZE = 50                 # 批处理大小
DOWNLOAD_MNIST = False          # 是否下载（未下载时需要填写下载）

# 准备训练集
train_data = torchvision.datasets.MNIST(            # 从torchvision.datasets.MNIST中下载数据
    root='./mnist',                                 # 数据存放的目录
    train=True,                                     # 下载的是用于训练的数据，训练集大小6w+,测试集大小1w+
    transform=torchvision.transforms.ToTensor(),    # 将原始数据转化为Tensor，便于Pytorch处理
    download=DOWNLOAD_MNIST                         # 是否下载
)
# 准备测试集
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False
)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[
         :2000] / 255.                              # 取前2000个数据作为测试集，并归一化
test_y = test_data.targets[:2000]                   # 取前2000个数据的标签作为x_y

# 批处理
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# define CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # 每个卷积层需包含Conv2d、ReLU、MaxPool2d
            nn.Conv2d(                      # 一个过滤器(filter)，高度代表提取的属性数量
                in_channels=1,              # 输入的数据高度，黑白图片的高度为1，RGB图片的高度是3
                out_channels=16,            # 过滤器个数，一个过滤器提取一种特征
                kernel_size=5,              # 过滤器的长宽大小(pix)
                stride=1,                   # 过滤器的步长
                padding=2      # 图片周围的像素点，一圈0，宽度为2，若想要conv2图片的宽度保持原来的宽度，padding = (kernel_size-1)/2
            ),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2)     # 在(2*2)pix的空间内提取最大值，作为这个区域的特征， output_shape (16,14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # output_shape (32,7*7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)    # 输出的图片大小为 32*7*7，分为10类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                   # (batch,32,7,7)
        x = x.view(x.size(0), -1)           # 将数据展平为 (batch,32*7*7)
        output = self.out(x)
        return output


cnn = CNN()                                             # 定义网络
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)    # 定义优化器
loss_func = nn.CrossEntropyLoss()                       # 定义损失函数

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()       # 梯度置零，防止叠加
        loss.backward()             # 计算梯度
        optimizer.step()            # 更新所有的参数

        if step % 50 == 0:  # 每 50步用测试集监测一下训练效果
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accurancy = sum(pred_y == test_y) / test_y.size(0)
            print('Step:', step, '| train loss %.4f' % loss.data.item(), '| test accurancy:', accurancy)

# print 10 prediction from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')